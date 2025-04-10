import gradio as gr
from datetime import datetime
from langchain_openai import ChatOpenAI
import pickle
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
import os
import pandas as pd
import tempfile
from langchain_community.tools import TavilySearchResults
from langchain_core.messages import AIMessage,AIMessageChunk,ToolMessage
from dotenv import load_dotenv

# TODO:
# -adding more model options

# --- CONFIGURATION ---
IS_DEBUG = 1
ENABLE_CHAT = 0
ENABLE_MILESTONES = 1
STUDY_PLANS_FILE = r'learning_plans/study_plans_data.pickle'
CURATED_MATERIALS_FILE = r'data/curated_additional_materials.txt'
LLM_MODEL="gpt-4o"
ENABLE_STREAM=1

# Hard-coded training timetable that determine the phase
TRAINING_PERIODS = {
    1: (datetime(2025, month=2, day=1, hour=6),datetime(2025, month=2, day=2, hour=6)),
    2: (datetime(2025, month=2, day=2, hour=6),datetime(2025, month=2, day=3, hour=6)),
    3: (datetime(2025, month=2, day=3, hour=6),datetime(2025, month=2, day=4, hour=6)),
    4: (datetime(2025, month=4, day=9, hour=6),datetime(2025, month=2, day=5, hour=6)),
}

# --- ENVIRONMENT SETUP ---
def setup_environment():

    os.environ["OPENAI_API_KEY"] = "DUMMY"
    os.environ["TAVILY_API_KEY"] = "DUMMY"

    if ENABLE_CHAT:
        """Load environment variables and setup initial state"""
        load_dotenv('.env')
        # Set environment variables from .env file
        #os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
        os.environ["TAVILY_API_KEY"] = os.getenv('TAVILY_API_KEY')

    # Determine current phase
    current_time = datetime.now()
    if IS_DEBUG==1:
        return 3  # Force phase 2 in debug mode
    else:
        for k,dates in TRAINING_PERIODS.items():
            if k==1 and (current_time<dates[0]):
                return 1
            elif k==4 and (current_time>dates[1]):
                return 4
            elif dates[0]<=current_time<dates[1]:
                return k
        raise Exception("BUG!")

# --- GLOBAL STATE ---
# Initialize global state
user_data = None
session_counter = 0
session_snapshots = []
print('loading study plans data...',end='')
user_datasets = pickle.load(open(STUDY_PLANS_FILE, 'rb')) # AT ROOT
print(f' done ({len(user_datasets)} items loaded)')
print('loading curated materials data...',end='')
additional_courses_data = pd.read_csv(CURATED_MATERIALS_FILE, sep='|', index_col=0)
additional_courses_data = (additional_courses_data['description'] + ' URL: ' + additional_courses_data['url']).to_list()
print(f' done ({len(additional_courses_data)} items)')
llm_options = {}
print('setting environment...',end='')
current_phase = setup_environment()
print(' done')
current_selected_phase = min(4,current_phase)
checkbox_elems = []

# --- TOOLS ---
@tool
def additional_materials_tool():
    """List of additional learning materials, such as videos, articles and courses, covering topics of teaching"""
    print('Obtaining learning materials')
    return additional_courses_data

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

@tool
def phase3_plan_tool():
    """Contains the personalized smart learning plan created for the student"""
    print('Obtaining smart_plan_phase3')
    return user_data['smart_plan_phase3']

@tool
def phase4_plan_tool():
    """Contains the personalized smart learning plan created for the student"""
    print('Obtaining smart_plan_phase4')
    return user_data['smart_plan_phase4']

@tool
def milestones_tool():
    """Contains the personalized milestones (max 10) created for the student"""
    print('Obtaining milestones')
    return user_data['milestones']

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
    global llm_options

    if ENABLE_CHAT:

        llm_options['config'] = {"configurable": {"thread_id": "1"}}
        llm_options['memory'] = MemorySaver()

        # Create tool lists based on settings
        common_tools=[]
        tools = {k:[] for k in range(1,5)}

        print('updating agents...',end='')
        if llm_options['use_search_tool']:
            print(' adding search tool ',end=' ')
            common_tools.append(initialize_search_tool())

        if llm_options['use_learningmaterial_tool']:
            print(' adding learning materials tool ', end=' ')
            common_tools.append(additional_materials_tool)

        if llm_options['use_milestones_tool']:
            print(' adding milestones tool ', end=' ')
            common_tools.append(milestones_tool)

        if llm_options['use_plan_tool']:
            print(' adding plan tool ', end=' ')
            tools[1].append(phase1_plan_tool)
            tools[2].append(phase2_plan_tool)
            tools[3].append(phase3_plan_tool)
            tools[4].append(phase4_plan_tool)

        # Create agents with appropriate tools
        llm_options['agent_phase1'] = create_agent(common_tools + tools[1])
        llm_options['agent_phase2'] = create_agent(common_tools + tools[2])
        llm_options['agent_phase3'] = create_agent(common_tools + tools[3])
        llm_options['agent_phase4'] = create_agent(common_tools + tools[4])

        print(' ...done')

def predict(message, history):
    """Handle message prediction using the appropriate agent"""
    global session_counter

    if len(history) == 0:
        if session_counter>0:
            snapshot = llm_options[f'agent_phase{current_selected_phase}'].get_state(llm_options['config'])
            session_snapshots.append(snapshot)
            update_agents() # to reset history
        session_counter+=1

    # Determine which agent to use based on current phase
    agent = llm_options[f'agent_phase{current_selected_phase}']

    # Invoke the agent
    if ENABLE_STREAM:
        response = ''
        for msg, metadata in agent.stream({"messages": [{"role": "user", "content": message}]},llm_options['config'],stream_mode="messages"):
            if isinstance(msg, AIMessageChunk) or isinstance(msg, AIMessage):
                response += msg.content
            elif isinstance(msg, ToolMessage):
                response += f'\n[used tool "{msg.name}"]\n'
            yield response
    else:
        response = agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            llm_options['config'],
            stream_mode="values",
        )
        response = response["messages"][-1].content

    return response

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
        if ENABLE_MILESTONES:
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

def authenticate(username, password):
    """Authenticate user with username and password"""
    global user_data

    if username in user_datasets and user_datasets[username]['password'] == password:
        user_data = user_datasets[username]
        user_data['username'] = username
        user_data['learning_state'] = load_user_state(username)

        if user_data['learning_state'] is None:
            user_data['learning_state'] = {
                'labels':user_data['milestones'],
                'states':[False]*len(user_data['milestones'])}
            save_user_state(username,user_data['learning_state'])
        else:
            assert len(user_data['learning_state']['labels'])==len(user_data['milestones'])
        reset_to_defaults()
        return True

    return False

def reset_to_defaults(*args):
    """Reset llm_options to default values"""
    global llm_options

    llm_options = {
        'system_prompt': user_data['assistant_prompt'].strip(),
        'use_plan_tool': 1,
        'use_search_tool': 1,
        'use_learningmaterial_tool': 1,
        'use_milestones_tool':1,
        'temperature': 0.3
    }
    return [
        llm_options['system_prompt'],
        llm_options['temperature'],
        llm_options['use_plan_tool'],
        llm_options['use_search_tool'],
        llm_options['use_learningmaterial_tool'],
        llm_options['use_milestones_tool']
    ]

# --- USER INTERFACE HELPERS ---
def save_pdf_file():
    """Save the current phase's PDF file to a temporary location"""
    file_name = f"UPBEAT_learning_plan_phase{current_selected_phase}"
    file_path = os.path.join(tempfile.gettempdir(), file_name)

    pdf_data_key = f"smart_plan_pdf_phase{current_selected_phase}"

    print(f'writing PDF file {file_path}...',end='')
    with open(file_path, 'wb') as pdf_file:
        pdf_file.write(user_data[pdf_data_key])
    print(' done')

    return file_path

def load_user_data(*args):
    """Load user greeting message based on selected phase"""
    user_name = user_data['data']['Q1. Full Name']
    return f"Hello **{user_name}**! Below is your personalized learning plan for the selected phase."

def load_smart_plan(*args):
    """Load the appropriate smart plan based on selected phase"""
    return user_data[f'smart_plan_phase{current_selected_phase}']

def get_current_date_message():
    """Generate a message with the current date and phase information"""
    current_date = datetime.now().strftime("%d.%m.%Y")

    if current_phase < 5:
        return f"**{current_date}<br>We're in phase {current_phase}.**"
    else:
        return f"**{current_date}<br>We're in past-training phase.**"

# --- SETTINGS PANEL FUNCTIONS ---
def apply_and_close(system_prompt, temperature, learning_plans_checkbox,
                    internet_search_checkbox,learning_material_checkbox,milestones_checkbox):
    """Apply settings and close the panel"""
    global llm_options

    llm_options['system_prompt'] = system_prompt.strip()
    llm_options['use_plan_tool'] = int(learning_plans_checkbox)
    llm_options['use_search_tool'] = int(internet_search_checkbox)
    llm_options['use_milestones_tool'] = int(milestones_checkbox)
    llm_options['use_learningmaterial_tool'] = int(learning_material_checkbox)
    llm_options['temperature'] = max(0,min(1.0, float(temperature)))

    update_agents()
    return gr.update(visible=False)

def close_no_changes():
    """Close settings panel without applying changes"""
    return gr.update(visible=False)

def toggle_milestones_panel():
    """Make the milestones panel visible"""
    return gr.update(visible=True)

def close_milestones_panel(*checkboxes):
    """Hide the milestones panel"""

    if user_data and "learning_state" in user_data:
        user_data['learning_state']['states'] = checkboxes
        save_user_state(user_data['username'],user_data['learning_state'])
    else:
        raise Exception("user_data has no learning_state attribute")

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
        llm_options['use_learningmaterial_tool'] > 0,
        llm_options['use_milestones_tool'] > 0
    )

# --- PHASE SWITCHING ---
def switch_to_onboarding():
    """Switch to onboarding phase"""
    global current_selected_phase
    current_selected_phase = 1
    return (
        gr.update(elem_classes=["phase-button", "green-button"]),
        gr.update(elem_classes=["phase-button", "gray-button"]),
        gr.update(elem_classes=["phase-button", "gray-button"]),
        gr.update(elem_classes=["phase-button", "gray-button"]),
        user_data['smart_plan_phase1'],
    )

def switch_to_phase2():
    """Switch to training phase"""
    global current_selected_phase
    current_selected_phase = 2
    return (
        gr.update(elem_classes=["phase-button", "gray-button"]),
        gr.update(elem_classes=["phase-button", "green-button"]),
        gr.update(elem_classes=["phase-button",  "gray-button"]),
        gr.update(elem_classes=["phase-button",  "gray-button"]),
        user_data['smart_plan_phase2'],
    )

def switch_to_phase3():
    """Switch to training phase"""
    global current_selected_phase
    current_selected_phase = 3
    return (
        gr.update(elem_classes=["phase-button", "gray-button"]),
        gr.update(elem_classes=["phase-button", "gray-button"]),
        gr.update(elem_classes=["phase-button",  "green-button"]),
        gr.update(elem_classes=["phase-button",  "gray-button"]),
        user_data['smart_plan_phase3'],
    )

def switch_to_phase4():
    """Switch to training phase"""
    global current_selected_phase
    current_selected_phase = 4
    return (
        gr.update(elem_classes=["phase-button", "gray-button"]),
        gr.update(elem_classes=["phase-button", "gray-button"]),
        gr.update(elem_classes=["phase-button",  "gray-button"]),
        gr.update(elem_classes=["phase-button",  "green-button"]),
        user_data['smart_plan_phase4'],
    )

def toggle_visibility(visible):
    """Toggle visibility of the smart plan"""
    return not visible, gr.update(visible=not visible)

# def load_milestone_data():
#     """Retrieve milestone tasks and their completion states from user_data"""
#     if user_data and "milestones" in user_data:
#         tasks = user_data["milestones"].get("tasks", [])
#         states = user_data["milestones"].get("states", [False] * len(tasks))  # Default: All unchecked
#         return tasks, states
#     return [], []

# --- MAIN UI DEFINITION ---
def create_chatbot_interface():
    """Create the Gradio interface for the chatbot"""
    css = """
    #sidebar {
        width: 120px !important;
        min-width: 120px !important;
        max-width: 120px !important;
        background-color: #f0f0f0 !important;
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
        margin-left: 120px; /* match sidebar width */
        padding: 20px;
        flex-grow: 1;
        min-width: 0;
        width: auto;
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
    
    /* Optional: Apply full flex layout to root if not already */
    .gradio-container > .gr-blocks {
        display: flex;
        flex-direction: row;
        width: 100%;
        height: 100%;
    }

    """
    global checkbox_elems

    with gr.Blocks(css=css) as app:
        rerender_trigger = gr.State(value=0)

        with gr.Row():
            # Sidebar
            with gr.Column(scale=0, min_width=150, elem_id="sidebar"):
                gr.HTML('<img src="/gradio_api/file=logo.png" style="width:100px; margin-bottom:20px;">')

                # Determine initial button states
                onboarding_initial = "green-button" if current_phase == 1 else "gray-button"
                phase2_initial = "green-button" if current_phase == 2 else "gray-button"
                phase3_initial = "green-button" if current_phase == 3 else "gray-button"
                phase4_initial = "green-button" if current_phase >= 4 else "gray-button"

                if ENABLE_CHAT:
                    options_button = gr.Button("Options")

                onboarding_btn = gr.Button("Onboarding", interactive=True,
                                           elem_classes=["phase-button", onboarding_initial])
                training1_btn = gr.Button("Phase 2",
                                         elem_classes=["phase-button", phase2_initial],
                                         interactive=(current_phase > 1))
                training2_btn = gr.Button("Phase 3",
                                         elem_classes=["phase-button", phase3_initial],
                                         interactive=(current_phase > 2))
                training3_btn = gr.Button("Phase 4",
                                         elem_classes=["phase-button", phase4_initial],
                                         interactive=(current_phase > 3))

                # Training phase message
                training_message = gr.Markdown(get_current_date_message())

            # Main Content
            with gr.Column(scale=1, elem_id="main-panel"):
                gr.Markdown("# UPBEAT Learning Assistant")
                greeting_textbox = gr.Markdown(value="Loading user data...")

                with gr.Row():
                    toggle_button = gr.Button("Show/Hide learning plan", elem_classes="plan-button")
                    if ENABLE_MILESTONES:
                        milestones_btn = gr.Button("Show/Hide milestones", elem_classes="plan-button")
                    download_btn = gr.Button("Save plan as PDF", elem_classes="plan-button")
                    download_btn_hidden = gr.DownloadButton(visible=False, elem_id="download_btn_hidden")

                smart_plan_markdown = gr.Markdown(visible=True)

                # Button click actions
                onboarding_btn.click(
                    switch_to_onboarding,
                    inputs=None,
                    outputs=[onboarding_btn, training1_btn,training2_btn,training3_btn,smart_plan_markdown]
                )
                training1_btn.click(
                    switch_to_phase2,
                    inputs=None,
                    outputs=[onboarding_btn, training1_btn,training2_btn,training3_btn,smart_plan_markdown]
                )

                training2_btn.click(
                    switch_to_phase3,
                    inputs=None,
                    outputs=[onboarding_btn, training1_btn,training2_btn,training3_btn,smart_plan_markdown]
                )

                training3_btn.click(
                    switch_to_phase4,
                    inputs=None,
                    outputs=[onboarding_btn, training1_btn,training2_btn,training3_btn,smart_plan_markdown]
                )

                visibility_state = gr.State(value=True)

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

                if ENABLE_CHAT:
                    gr.Markdown(
                        "Chatbot below can assist you with your studies. You can discuss about your learning plan or search information.")

                    chat = gr.ChatInterface(
                        predict,
                        type="messages",
                        title="Learning Assistant Bot",
                        examples=[
                            "Hi! Who are you and what can you do for me?",
                            "Help me in getting started with my learning!",
                            "What are my top learning priorities?",
                            "Give me a summary what you know about me and my priorities.",
                            "What are essential skills I need to concentrate in?",
                            "What tools you have available?"
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
                @gr.render(inputs=[rerender_trigger])
                def render_checkboxes(trigger):
                    global user_data
                    """Dynamically generates checkboxes based on user login."""
                    # Dynamically creating checkboxes
                    if user_data is None:
                        checkboxes = []
                    else:
                        data = user_data['learning_state']
                        checkboxes = [gr.Checkbox(label=data['labels'][k], value=data['states'][k]) for k in
                                      range(len(data['labels']))]

                    close_milestones_btn = gr.Button("Close", elem_classes="plan-button")
                    # Connecting button click to process function
                    close_milestones_btn.click(
                        close_milestones_panel,inputs=checkboxes,outputs=[milestones_panel]
                    )

            if ENABLE_MILESTONES:
                milestones_btn.click(
                    toggle_milestones_panel,
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
                gr.Markdown("**TOOLS that LLM can access**")
                with gr.Row():
                    learning_plans_checkbox = gr.Checkbox(label="Learning plan")
                    internet_search_checkbox = gr.Checkbox(label="Internet search")
                    learning_material_checkbox = gr.Checkbox(label="Learning materials")
                    milestones_checkbox = gr.Checkbox(label="Milestones")

                reset_settings_button = gr.Button("Reset settings to default")

                # Apply and close button
                with gr.Row():
                    apply_all_button = gr.Button("Apply all & close")
                    close_no_changes_button = gr.Button("Close without changes")

                if ENABLE_CHAT:
                    # Button actions
                    options_button.click(
                        toggle_settings_panel,
                        outputs=[
                            settings_panel,
                            system_prompt,
                            temperature,
                            learning_plans_checkbox,
                            internet_search_checkbox,
                            learning_material_checkbox,
                            milestones_checkbox
                        ]
                    )

                    reset_settings_button.click(
                        reset_to_defaults,
                        inputs=None,
                        outputs=[
                            system_prompt,
                            temperature,
                            learning_plans_checkbox,
                            internet_search_checkbox,
                            learning_material_checkbox,
                            milestones_checkbox
                        ]
                    )
                    apply_all_button.click(
                        apply_and_close,
                        inputs=[
                            system_prompt,
                            temperature,
                            learning_plans_checkbox,
                            internet_search_checkbox,
                            learning_material_checkbox,
                            milestones_checkbox
                        ],
                        outputs=settings_panel
                    )
                    close_no_changes_button.click(close_no_changes, outputs=settings_panel)
        app.load(lambda x: x, inputs=[rerender_trigger], outputs=[rerender_trigger])
    return app

# --- MAIN EXECUTION ---
def main():
    """Main function to launch the application"""
    # Create the interface
    demo = create_chatbot_interface()

    # Launch the app with appropriate settings
    if IS_DEBUG>0:
        IDs = list(user_datasets.keys())
        ind = 0
        print(f'!! DEBUG MODE ({IS_DEBUG}): Auto-login as {IDs[ind]} !!')
        authenticate(username=IDs[ind], password=user_datasets[IDs[ind]]["password"])
        update_agents()
        demo.launch(share=False, allowed_paths=["logo.png"])
    else:
        update_agents()
        demo.launch(auth=authenticate, share=False, allowed_paths=["logo.png"])

if __name__ == "__main__":
    main()