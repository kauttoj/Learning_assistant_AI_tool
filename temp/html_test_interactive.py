import gradio as gr
import json

# Python function to process the checkbox states from the hidden textbox.
def process_checkbox_states(states_json):
    try:
        states = json.loads(states_json)
    except Exception as e:
        states = {}
    print("Checkbox states from JS:", states)
    return f"Checkbox states: {states}"

# Custom HTML with checkboxes embedded within text and a button that triggers JS reading of the states.
html_content = """
<div style="font-family: Arial, sans-serif; font-size: 16px;">
  <p>
    Please select your options:
    <label style="margin: 0 10px;">
      <input type="checkbox" id="checkbox1"> Option 1
    </label>
    <label style="margin-left: 20px;">
      <input type="checkbox" id="checkbox2"> Option 2
    </label>
  </p>
  <p>
    <button id="read_state">Read Checkbox States</button>
  </p>
</div>
"""

# JavaScript to be injected. It listens for the click on the in-HTML button and writes the checkbox states to the hidden textbox.
js_code = """
function init_custom_js() {
  const btn = document.querySelector("#read_state");
  if (btn) {
    btn.addEventListener("click", function() {
      const cb1 = document.querySelector("#checkbox1") || document.querySelector("#checkbox1"); // fallback in case of typo
      const cb2 = document.querySelector("#checkbox2");
      // Build a JSON string with the states of the checkboxes.
      const states = {
          "Option 1": document.querySelector("#checkbox1") ? document.querySelector("#checkbox1").checked : null,
          "Option 2": cb2 ? cb2.checked : null
      };
      // Write the state JSON to the hidden textbox.
      const hidden_input = document.querySelector("#hidden_states");
      if (hidden_input) {
          hidden_input.value = JSON.stringify(states);
          // Optionally, trigger a change event if needed.
          hidden_input.dispatchEvent(new Event("change"));
      }
      console.log("JS: Updated hidden input with states:", states);
    }
init_custom_js = function() {
  // Make sure the DOM is fully loaded.
  document.addEventListener("DOMContentLoaded", function() {
    // Attach event listener to the button
    const btn = document.querySelector("#read_state");
    if (btn) {
      btn.addEventListener("click", function() {
          // Read the state of checkboxes and update hidden input
          const cb1 = document.querySelector("#checkbox1");
          const cb2 = document.querySelector("#checkbox2");
          let states = {
              "Option 1": cb1 ? cb1.checked : null,
              "Option 2": cb2 ? cb2.checked : null
          };
          const hidden_input = document.querySelector("#hidden_states");
          if (hidden_input) {
              hidden_input.value = JSON.stringify(states);
              hidden_input.dispatchEvent(new Event("change"));
          }
          console.log("JS: Updated hidden input with states:", states);
        };
  // Attach event listener
  const button = document.querySelector("#read_state");
  if (button) {
      button.addEventListener("click", init_custom_js);
  }
};
// Call the setup function immediately.
init_custom_js();
"""

with gr.Blocks() as demo:
    gr.Markdown("### Custom HTML Checkboxes with JS Injection")
    # Insert the custom HTML into the interface.
    gr.HTML(html_content)
    # A hidden component to pass data from JS to Python.
    hidden_states = gr.Textbox(label="Hidden States", visible=False, elem_id="hidden_states")
    # The Process button reads the value from the hidden textbox.
    process_btn = gr.Button("Process")
    output = gr.Textbox(label="Output")
    process_btn.click(process_checkbox_states, inputs=[hidden_states], outputs=[output])
    # Inject JavaScript via the js parameter.
    demo.load(js=js_code)

demo.launch()
