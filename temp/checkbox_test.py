import gradio as gr

data = None
# Function to process and display selected checkbox values
def process_and_save(*choices):
    selected_options = [i for i,x in enumerate(choices,start=1) if x]
    print(f"Selected Options: {selected_options}")

def make_GUI():
    global data
    # Creating the Gradio Blocks context
    with gr.Blocks() as demo:
        rerender_trigger = gr.State(value=0)

        gr.Markdown("### Select options and save")
        @gr.render(inputs=[rerender_trigger])  # No external inputs required for rendering
        def create_checkboxes(trigger):

            print(f'running create_checkboxes(), trigger={trigger}, data={data}')
            # Dynamically creating checkboxes
            if data is None:
                checkboxes=[]
            else:
                checkboxes = [gr.Checkbox(label=data['labels'][k],value=data['states'][k]) for k in range(len(data['labels']))]

            save_button = gr.Button("Save")

            # Connecting button click to process function
            save_button.click(process_and_save, inputs=checkboxes, outputs=[])

        demo.load(lambda x: x, inputs=[rerender_trigger], outputs=[rerender_trigger])

    return demo

demo = make_GUI()

data = {'labels':['option 1','option 2','option 3'],'states':[True,False,False]}
# Launch the Gradio interface
demo.launch()
