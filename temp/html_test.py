import gradio as gr

# Custom HTML with improved checkbox styling
custom_html = """
<div id="survey-form" style="background-color: #f5f7fa; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
  <h2 style="color: #1a5fb4; margin-bottom: 15px;">User Experience Survey</h2>

  <div class="question" style="margin-bottom: 15px;">
    <p style="font-weight: bold; margin-bottom: 8px;">What features do you use most? (select all that apply)</p>
    <div style="margin-left: 15px;">
      <label class="checkbox-container" style="display: block; margin-bottom: 8px; cursor: pointer;">
        <input type="checkbox" name="features" value="AI generation" id="feature-ai" class="survey-checkbox"> 
        <span class="checkbox-text">AI content generation</span>
      </label>
      <label class="checkbox-container" style="display: block; margin-bottom: 8px; cursor: pointer;">
        <input type="checkbox" name="features" value="Data visualization" id="feature-viz" class="survey-checkbox"> 
        <span class="checkbox-text">Data visualization</span>
      </label>
      <label class="checkbox-container" style="display: block; margin-bottom: 8px; cursor: pointer;">
        <input type="checkbox" name="features" value="Image processing" id="feature-img" class="survey-checkbox"> 
        <span class="checkbox-text">Image processing</span>
      </label>
      <label class="checkbox-container" style="display: block; margin-bottom: 8px; cursor: pointer;">
        <input type="checkbox" name="features" value="Audio analysis" id="feature-audio" class="survey-checkbox"> 
        <span class="checkbox-text">Audio analysis</span>
      </label>
    </div>
  </div>

  <div class="question" style="margin-bottom: 15px;">
    <p style="font-weight: bold; margin-bottom: 8px;">How did you discover our app? (select all that apply)</p>
    <div style="margin-left: 15px;">
      <label class="checkbox-container" style="display: block; margin-bottom: 8px; cursor: pointer;">
        <input type="checkbox" name="discovery" value="Search engine" id="disc-search" class="survey-checkbox"> 
        <span class="checkbox-text">Search engine</span>
      </label>
      <label class="checkbox-container" style="display: block; margin-bottom: 8px; cursor: pointer;">
        <input type="checkbox" name="discovery" value="Social media" id="disc-social" class="survey-checkbox"> 
        <span class="checkbox-text">Social media</span>
      </label>
      <label class="checkbox-container" style="display: block; margin-bottom: 8px; cursor: pointer;">
        <input type="checkbox" name="discovery" value="Friend recommendation" id="disc-friend" class="survey-checkbox"> 
        <span class="checkbox-text">Friend recommendation</span>
      </label>
      <label class="checkbox-container" style="display: block; margin-bottom: 8px; cursor: pointer;">
        <input type="checkbox" name="discovery" value="Blog or article" id="disc-blog" class="survey-checkbox"> 
        <span class="checkbox-text">Blog or article</span>
      </label>
    </div>
  </div>

</div>

"""

# JavaScript to handle checkboxes and form submission
js = """
// Run after DOM is fully loaded
document.addEventListener('DOMContentLoaded', function() {
  // Prevent Gradio from interfering with checkbox states
  const checkboxes = document.querySelectorAll('.survey-checkbox');

  checkboxes.forEach(function(checkbox) {
    checkbox.addEventListener('click', function(e) {
      // Prevent event propagation to stop Gradio from handling this event
      e.stopPropagation();
    });
  });
});
"""

# Custom CSS to fix the checkbox appearance
css = """
/* Make checkboxes appear normal */
.survey-checkbox {
  appearance: auto !important;
  -webkit-appearance: checkbox !important;
  -moz-appearance: checkbox !important;
  opacity: 1 !important;
  width: auto !important;
  height: auto !important;
  margin-right: 8px !important;
  position: relative !important;
  z-index: 1 !important;
  visibility: visible !important;
}

/* Style the checkbox container on hover */
.checkbox-container:hover {
  background-color: rgba(0, 0, 0, 0.05);
  border-radius: 4px;
  padding-left: 5px;
  margin-left: -5px;
}

/* Style the submit button hover */
#submit-survey:hover {
  background-color: #0d4d96;
}
"""


def process_survey(features, discovery):
    """Process the survey results"""
    features_text = features if features else "None selected"
    discovery_text = discovery if discovery else "None selected"
    return f"Thank you for your feedback!\n\nSelected features: {features_text}\n\nDiscovery channels: {discovery_text}"


# Create Gradio app with HTML and custom CSS/JS
with gr.Blocks(css=css, js=js) as demo:
    with gr.Row():
        with gr.Column(scale=2):
            # Display the HTML form
            gr.HTML(custom_html)

            # Hidden textboxes to receive values from JavaScript
            features_output = gr.Textbox(elem_id="features-output", visible=False)
            discovery_output = gr.Textbox(elem_id="discovery-output", visible=False)

            # Response area that appears after submission
            response = gr.Textbox(elem_id="gradio-response", label="Survey Response", visible=False)

            gr.Button('Process')

    # Update the response when hidden fields are updated
    features_output.change(
        process_survey,
        inputs=[features_output, discovery_output],
        outputs=response
    )

# Launch the app
demo.launch()