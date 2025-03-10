import gradio as gr
from datetime import datetime
from langchain_openai import ChatOpenAI
import pickle
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
import os
import tempfile
from langchain_community.tools import TavilySearchResults
from dotenv import load_dotenv

# TODO:
# -Load/save milestone status from disk
# -Update milestones when phase is selected

# --- CONFIGURATION ---
IS_DEBUG = 1
STUDY_PLANS_FILE = 'study_plans_data.pickle'
LLM_MODEL="gpt-4o-mini"
TRAINING_PERIOD_START = datetime(2025, month=4, day=1, hour=6)
TRAINING_PERIOD_END = datetime(2025, month=4, day=21, hour=23)

# --- ENVIRONMENT SETUP ---
def setup_environment():
    """Load environment variables and setup initial state"""
    load_dotenv('.env')

    # Set environment variables from .env file
    os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    os.environ["GOOGLE_CSE_ID"] = os.getenv("GOOGLE_CSE_ID")
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_SEARCH_API")
    os.environ["TAVILY_API_KEY"] = os.getenv('TAVILY_API_KEY')

    # Determine current phase
    current_time = datetime.now()
    if IS_DEBUG:
        return 2  # Force phase 2 in debug mode
    elif current_time < TRAINING_PERIOD_START:
        return 1
    elif current_time < TRAINING_PERIOD_END:
        return 2
    else:
        return 3

# --- GLOBAL STATE ---
# Initialize global state
user_data = None
user_datasets = None
llm_options = {}
current_phase = setup_environment()
current_selected_phase = current_phase

# --- TOOLS ---
@tool
def phase1_plan_tool():
    """Contains the personalized smart learning plan created for the student"""
    print('Obtaining smart_plan_phase1')
    return user_data['smart_plan_phase1']


@tool
def phase2_plan_tool():
    """Contains the personalized smart learning plan created for the student"""
    print('Obtaining smart_plan_phase2')
    return user_data['smart_plan_phase2']

def initialize_search_tool():
    """Initialize the search tool"""
    return TavilySearchResults(
        max_results=5,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=True,
        include_images=False
    )

# --- LLM FUNCTIONS ---
def create_agent(tools_list):
    """Creates a new agent with the given tools and memory."""
    model = ChatOpenAI(model=LLM_MODEL,temperature=llm_options.get('temperature', 0.30))
    return create_react_agent(
        model,
        tools=tools_list,
        prompt=llm_options['system_prompt'],
        checkpointer=llm_options['memory']
    )


def update_agents():
    """Update the LLM agents based on current options"""
    llm_options['config'] = {"configurable": {"thread_id": "1"}}
    llm_options['memory'] = MemorySaver()

    # Create tool lists based on settings
    phase1_tools = []
    phase2_tools = []

    if llm_options['use_search_tool']:
        search_tool = initialize_search_tool()
        phase1_tools.append(search_tool)
        phase2_tools.append(search_tool)

    if llm_options['use_plan_tool']:
        phase1_tools.append(phase1_plan_tool)
        phase2_tools.append(phase2_plan_tool)

    # Create agents with appropriate tools
    llm_options['agent_phase1'] = create_agent(phase1_tools)
    llm_options['agent_phase2'] = create_agent(phase2_tools)

def predict(message, history):
    """Handle message prediction using the appropriate agent"""
    if len(history) == 0:
        print('...empty history, creating agents')
        update_agents()

    # Determine which agent to use based on current phase
    agent = llm_options['agent_phase1'] if current_phase == 1 else llm_options['agent_phase2']

    # Get current state and debug info
    snapshot = agent.get_state(llm_options['config'])
    snapshot_size = len(snapshot.values['messages']) if len(snapshot.values) else 0
    print(f"current history size: Gradio {len(history)}, Agent {snapshot_size}")

    # Invoke the agent
    response = agent.invoke(
        {"messages": [{"role": "user", "content": message}]},
        llm_options['config'],
        stream_mode="values",
    )

    return response["messages"][-1].content

# --- USER DATA & AUTHENTICATION ---
# Add these functions for user state management
def ensure_user_directory(username):
    """Create user directory if it doesn't exist"""
    user_dir = os.path.join('user_data', username)
    if not os.path.exists(user_dir):
        os.makedirs(user_dir, exist_ok=True)
    return user_dir

def load_user_state(username):
    """Load user state from pickle file or create default if not exists"""
    user_dir = ensure_user_directory(username)
    state_file = os.path.join(user_dir, 'state_variables.pickle')

    if os.path.exists(state_file):
        try:
            with open(state_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            raise(Exception(f"Error loading user state: {e}"))
    else:
        return None

def save_user_state(username, state):
    """Save user state to pickle file using a safe temp file method"""
    user_dir = ensure_user_directory(username)
    state_file = os.path.join(user_dir, 'state_variables.pickle')
    temp_file = os.path.join(user_dir, 'state_variables.temp.pickle')

    try:
        # First write to a temporary file
        with open(temp_file, 'wb') as f:
            pickle.dump(state, f)
            f.flush()  # Ensure data is written to disk
            os.fsync(f.fileno())  # Force write to physical storage

        # Then rename the temp file to the target file (atomic operation)
        if os.path.exists(state_file):
            # On Windows, we need to remove the target file first
            if os.name == 'nt' and os.path.exists(state_file):
                os.remove(state_file)

        os.rename(temp_file, state_file)
        return True
    except Exception as e:
        print(f"Error saving user state: {e}")
        # Clean up temp file if it exists after an error
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass
        return False

def load_user_datasets():
    """Load user datasets from pickle file"""
    global user_datasets
    user_datasets = pickle.load(open(STUDY_PLANS_FILE, 'rb'))

def authenticate(username, password):
    """Authenticate user with username and password"""
    global user_data

    # Ensure user_datasets is loaded
    if user_datasets is None:
        load_user_datasets()

    if username in user_datasets and user_datasets[username]['password'] == password:
        user_data = user_datasets[username]
        user_data['learning_state'] = load_user_state(username)

        if user_data['learning_state'] is None:
            user_data['learning_state']={}
            user_data['learning_state'][1] = {
                'labels':user_data['milestones_phase1'],
                'states':[False]*len(user_data['milestones_phase1'])}
            user_data['learning_state'][2] = {
                'labels': user_data['milestones_phase2'],
                'states': [False] * len(user_data['milestones_phase2'])
            }
            save_user_state(username,user_data['learning_state'])
        else:
            assert len(user_data['learning_state'][1]['labels'])==len(user_data['milestones_phase1'])
            assert len(user_data['learning_state'][2]['labels']) == len(user_data['milestones_phase2'])
        reset_to_defaults()
        return True

    return False

def reset_to_defaults():
    """Reset llm_options to default values"""
    global llm_options

    llm_options = {
        'system_prompt': user_data['assistant_prompt'].strip(),
        'use_plan_tool': 1,
        'use_search_tool': 1,
        'use_learningmaterial_tool': 1,
        'temperature': 0.3
    }

# --- USER INTERFACE HELPERS ---
def save_pdf_file():
    """Save the current phase's PDF file to a temporary location"""
    file_name = "UPBEAT_onboarding_plan.pdf" if current_selected_phase == 1 else "UPBEAT_training_plan.pdf"
    file_path = os.path.join(tempfile.gettempdir(), file_name)

    pdf_data_key = 'smart_plan_pdf_phase1' if current_selected_phase == 1 else 'smart_plan_pdf_phase2'

    with open(file_path, 'wb') as pdf_file:
        pdf_file.write(user_data[pdf_data_key])

    return file_path


def load_user_data(_):
    """Load user greeting message based on selected phase"""
    user_name = user_data['data']['Q1. Full Name']

    if current_selected_phase == 1:
        return f"Hello **{user_name}**! Below is you personalized UPBEAT learning plan for onboarding phase."
    else:
        return f"Hello **{user_name}**! Below is you personalized UPBEAT learning plan for training phase. You can go back to onboarding phase plan if you need to."


def load_smart_plan(_):
    """Load the appropriate smart plan based on selected phase"""
    return user_data['smart_plan_phase1'] if current_selected_phase == 1 else user_data['smart_plan_phase2']


def get_current_date_message():
    """Generate a message with the current date and phase information"""
    current_date = datetime.now().strftime("%d.%m.%Y")

    if current_phase == 1:
        return f"**{current_date}<br>We're in onboarding phase.**"
    elif current_phase == 2:
        return f"**{current_date}<br>We're in training phase.**"
    else:
        return f"**{current_date}<br>We're in past-training phase.**"


# --- SETTINGS PANEL FUNCTIONS ---
def apply_and_close(system_prompt, temperature, learning_plans_checkbox,
                    internet_search_checkbox, learning_material_checkbox):
    """Apply settings and close the panel"""
    global llm_options

    llm_options['system_prompt'] = system_prompt.strip()
    llm_options['use_plan_tool'] = int(learning_plans_checkbox)
    llm_options['use_search_tool'] = int(internet_search_checkbox)
    llm_options['use_learningmaterial_tool'] = int(learning_material_checkbox)
    llm_options['temperature'] = max(0, min(1.0, float(temperature)))

    update_agents()
    return gr.update(visible=False)


def close_no_changes():
    """Close settings panel without applying changes"""
    return gr.update(visible=False)

def toggle_milestones_panel():
    """Make the milestones panel visible"""
    return gr.update(visible=True)

def close_milestones_panel(checkbox_values):
    """Hide the milestones panel"""

    if user_data and "learning_state" in user_data:
        user_data['learning_state'][current_selected_phase]['states'] = checkbox_values
        save_user_state(user_data['data']['Q1. Full Name'], user_data['learning_state'])

    return gr.update(visible=False)

def get_checkbox_values(*checkboxes):
    """Extracts the latest values from dynamically created checkboxes"""
    return list(checkboxes)

def toggle_settings_panel():
    """Toggle the visibility of the settings panel"""
    return [gr.update(visible=True)] + list(update_options_panel())

def update_options_panel():
    """Update the settings panel with current values"""
    return (
        llm_options['system_prompt'],
        str(llm_options['temperature']),
        llm_options['use_plan_tool'] > 0,
        llm_options['use_search_tool'] > 0,
        llm_options['use_learningmaterial_tool'] > 0
    )

# --- PHASE SWITCHING ---
def switch_to_onboarding():
    """Switch to onboarding phase"""
    global current_selected_phase
    current_selected_phase = 1
    return (
        gr.update(elem_classes=["phase-button", "green-button"]),
        gr.update(elem_classes=["phase-button", "gray-button"]),
        user_data['smart_plan_phase1'],
    )

def switch_to_training():
    """Switch to training phase"""
    global current_selected_phase
    current_selected_phase = 2
    return (
        gr.update(elem_classes=["phase-button", "gray-button"]),
        gr.update(elem_classes=["phase-button", "green-button"]),
        user_data['smart_plan_phase2'],
    )

def toggle_visibility(visible):
    """Toggle visibility of the smart plan"""
    return not visible, gr.update(visible=not visible)

def load_milestone_data():
    """Retrieve milestone tasks and their completion states from user_data"""
    if user_data and "milestones" in user_data:
        tasks = user_data["milestones"].get("tasks", [])
        states = user_data["milestones"].get("states", [False] * len(tasks))  # Default: All unchecked
        return tasks, states
    return [], []

def update_milestone_checkboxes():
    """Dynamically update milestone checkboxes after GUI is launched"""

    task_texts, task_states = ['test1','afsa afasffssasfafasf'],[True,False]
    checkboxes = [gr.Checkbox(label=task_texts[i], value=task_states[i]) for i in range(len(task_texts))]
    return checkboxes

# --- MAIN UI DEFINITION ---
def create_chatbot_interface():
    """Create the Gradio interface for the chatbot"""
    css = """
    #sidebar {
        width: 120px !important;
        min-width: 120px !important;
        max-width: 120px !important;
        background-color: #f0f0f0;
        display: flex;
        flex-direction: column;
        align-items: center;
        height: 100vh;
        position: fixed;
        left: 0;
        top: 0;
        bottom: 0;
        padding-top: 20px;
    }
    #main-panel {
        margin-left: 50px;
        width: calc(100% - 160px);
        padding: 20px;
    }
    #settings-panel {
        position: absolute;
        top: 20px;
        left: 0;
        width: 100% !important;
        min-width: unset !important;
        background: white;
        padding: 15px;
        border: 2px solid #ccc;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.2);
        z-index: 1000;
    }    
    #milestones-panel {
        position: absolute;
        top: 40px;
        left: 0;
        width: 100% !important;
        min-width: 350px !important;
        max-width: 600px !important;
        background: white;
        padding: 25px;
        border: 2px solid #ccc;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.2);
        z-index: 1000;
    }        
    #system-prompt-box textarea {
        overflow-y: auto !important;
        resize: vertical;
        max-height: 300px;
    }
    .plan-button {
        min-width: 165px;
        max-width: 165px;
        padding: 10px 20px;
        font-size: 14px;
        white-space: nowrap;
    }        
    .green-button {
        background-color: #4CAF50;
        color: white;
        border-color: #4CAF50;
        border: 3px solid black !important;
        padding: 10px;
        font-size: 16px;        
    }
    .gray-button {
        background-color: #f0f0f0;
        color: black;
        border-color: #ccc;
        border: 3px solid black !important;
        padding: 10px;
        font-size: 16px;        
    }
    """

    with gr.Blocks(css=css) as app:
        checkbox_data = gr.State({'labels': [], 'states': []})

        with gr.Row():
            # Sidebar
            with gr.Column(scale=0, min_width=150, elem_id="sidebar"):
                gr.HTML('<img src="/gradio_api/file=logo.png" style="width:100px; margin-bottom:20px;">')

                # Determine initial button states
                onboarding_initial = "green-button" if current_phase == 1 else "gray-button"
                training_initial = "green-button" if current_phase == 2 else "gray-button"

                options_button = gr.Button("Options")
                onboarding_btn = gr.Button("Onboarding", interactive=True,
                                           elem_classes=["phase-button", onboarding_initial])
                training_btn = gr.Button("Training",
                                         elem_classes=["phase-button", training_initial],
                                         interactive=(current_phase > 1))

                # Training phase message
                training_message = gr.Markdown(get_current_date_message())

            # Main Content
            with gr.Column(scale=1, elem_id="main-panel"):
                gr.Markdown("# UPBEAT Learning Assistant")
                greeting_textbox = gr.Markdown(value="Loading user data...")

                with gr.Row():
                    toggle_button = gr.Button("Show/Hide learning plan", elem_classes="plan-button")
                    milestones_btn = gr.Button("Show/Hide milestones", elem_classes="plan-button")
                    download_btn = gr.Button("Save plan as PDF", elem_classes="plan-button")
                    download_btn_hidden = gr.DownloadButton(visible=False, elem_id="download_btn_hidden")

                smart_plan_markdown = gr.Markdown(visible=False)

                # Button click actions
                onboarding_btn.click(
                    switch_to_onboarding,
                    inputs=None,
                    outputs=[onboarding_btn, training_btn, smart_plan_markdown]
                )
                training_btn.click(
                    switch_to_training,
                    inputs=None,
                    outputs=[onboarding_btn, training_btn, smart_plan_markdown]
                )

                visibility_state = gr.State(value=False)

                toggle_button.click(
                    toggle_visibility,
                    inputs=visibility_state,
                    outputs=[visibility_state, smart_plan_markdown]
                )

                download_btn.click(
                    fn=save_pdf_file,
                    inputs=None,
                    outputs=[download_btn_hidden]
                ).then(
                    fn=None, inputs=None, outputs=None,
                    js="() => document.querySelector('#download_btn_hidden').click()"
                )

                gr.Markdown(
                    "Chatbot below can assist you with your studies. You can discuss about your learning plan or search information.")

                chat = gr.ChatInterface(
                    predict,
                    type="messages",
                    title="Learning Assistant Bot",
                    examples=[
                        "Hi! Who are you and what can you do for me?",
                        "Help me in getting started with my learning",
                        "What are my top learning priorities?",
                        "What are essential skills I need to learn during onboarding?"
                    ],
                    save_history=True,
                    run_examples_on_click=True
                )

                app.load(load_user_data, inputs=None, outputs=[greeting_textbox])
                app.load(load_smart_plan, inputs=None, outputs=[smart_plan_markdown])

            # Milestones
            with gr.Column(scale=0, min_width=300, visible=False, elem_id="milestones-panel") as milestones_panel:
                gr.Markdown("## My milestones  \nThese are tasks related to your learning plan. Tick when completed.")

                # Task checkboxes
                @gr.render(inputs=checkbox_data)
                def render_checkboxes(data):
                    """Dynamically generates checkboxes based on user login."""
                    checkboxes = []
                    if 'learning_state' in data:
                        labels, states = data['learning_state'][current_selected_phase]['labels'],data['learning_state'][current_selected_phase]['states']
                    else:
                        labels, states = [],[]

                    with gr.Column():
                        for label, state in zip(labels, states):
                            checkboxes.append(gr.Checkbox(label=label, value=state))

                    return checkboxes  # Not necessary, but for clarity

                close_milestones_btn = gr.Button("Close", elem_classes="plan-button")

            milestones_btn.click(
                toggle_milestones_panel,
                outputs=[milestones_panel]
            )

            close_milestones_btn.click(
                close_milestones_panel,
                inputs=[checkbox_data],
                outputs=[milestones_panel]
            )

            # Settings Panel
            with gr.Column(scale=0, min_width=350, visible=False, elem_id="settings-panel") as settings_panel:
                gr.Markdown("### Settings Panel")

                # System Prompt Section
                system_prompt = gr.Textbox(
                    label="system_prompt",
                    lines=10,
                    max_lines=25,
                    value="...",
                    elem_id="system-prompt-box"
                )

                # Temperature setting
                with gr.Row():
                    temperature = gr.Textbox(label="temperature", value="0.7")

                # LLM Tools selection
                with gr.Row():
                    gr.Markdown("**LLM tools:**")
                    learning_plans_checkbox = gr.Checkbox(label="Learning Plans")
                    internet_search_checkbox = gr.Checkbox(label="Internet Search")
                    learning_material_checkbox = gr.Checkbox(label="Learning materials", visible=False)

                reset_settings_button = gr.Button("Reset settings to default")

                # Apply and close button
                with gr.Row():
                    apply_all_button = gr.Button("Apply all & close")
                    close_no_changes_button = gr.Button("Close without changes")

                # Button actions
                options_button.click(
                    toggle_settings_panel,
                    outputs=[
                        settings_panel,
                        system_prompt,
                        temperature,
                        learning_plans_checkbox,
                        internet_search_checkbox,
                        learning_material_checkbox
                    ]
                )

                reset_settings_button.click(
                    reset_to_defaults,
                    outputs=[
                        system_prompt,
                        temperature,
                        learning_plans_checkbox,
                        internet_search_checkbox,
                        learning_material_checkbox
                    ]
                )
                apply_all_button.click(
                    apply_and_close,
                    inputs=[
                        system_prompt,
                        temperature,
                        learning_plans_checkbox,
                        internet_search_checkbox,
                        learning_material_checkbox
                    ],
                    outputs=settings_panel
                )
                close_no_changes_button.click(close_no_changes, outputs=settings_panel)

        def update_ui():
            """Updates UI state when a user logs in."""
            return user_data  # Update state with user-specific checkbox data

        # Call update_ui to refresh UI after login
        app.load(update_ui, inputs=[], outputs=[checkbox_data])

    return app

# --- MAIN EXECUTION ---
def main():
    """Main function to launch the application"""
    # Load user datasets
    load_user_datasets()

    # Create the interface
    demo = create_chatbot_interface()

    # Launch the app with appropriate settings
    if IS_DEBUG:
        IDs = list(user_datasets.keys())
        ind = 1
        authenticate(username=IDs[ind], password=user_datasets[IDs[ind]]["password"])
        demo.launch(share=False, allowed_paths=["logo.png"])
    else:
        demo.launch(auth=authenticate, share=False, allowed_paths=["logo.png"])

if __name__ == "__main__":
    main()