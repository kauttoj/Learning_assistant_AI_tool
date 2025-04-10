import os
import pickle
import io
import time
import re
import random
import ast
import numpy as np
import pandas as pd
import base64
import markdown2
from datetime import datetime
from xhtml2pdf import pisa
from PIL import Image, ImageDraw
from dotenv import load_dotenv
from openai import OpenAI
from litellm import completion

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv('.env')
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# -----------------------------
# LLM Configurations
# -----------------------------
LOCAL_MODEL = 1
if LOCAL_MODEL:
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
    MODEL_STR = 'deepthinkers-phi4'  # "mradermacher/DeepThinkers-Phi4-GGUF"
    llm_config_large = {
        "model": MODEL_STR,
        "max_tokens": 10000,
        "temperature": 0.1,
    }
    llm_config_small = {
        "model": MODEL_STR,
        "temperature": 0.1,
        "max_tokens": 10000,
    }
else:
    llm_config_large = {
        "model": 'claude-3-7-sonnet-latest',
        "max_tokens": 16384,
        "temperature": 0.3,
        "thinking": {"type": "enabled", "budget_tokens": 5000}
    }
    llm_config_small = {
        "model": 'gpt-4o',
        "temperature": 0.0,
        "max_tokens": 16384
    }
    # llm_config_large=llm_config_small
# -----------------------------
# Prompt Templates
# -----------------------------
ending_text = '__We are glad to have you onboard :) If you have any questions, please contact teachers.__'

PHASE1_PROMPT_TEMPLATE_BEGINNER = '''
# Smart learning plan (onboarding)

Dear {student_name},

Thank you for participating in our Entrepreneurship Training Course!
Below is your plan to build the foundational skills you currently rate as beginner.

## Essential learning topics and materials

{beginner_level_materials}

{ending_text}
'''

PHASE1_PROMPT_TEMPLATE_ADVANCED = ''' 
# ROLE # 

You are a teacher tasked with creating a personalized Smart Learning Plan (SLP) for a student who already possesses basic skills in all training topics.  

# CONTEXT # 

The training session topics are divided into four (4) core modules. Each module has objectives and some pre-defined assignments. This is shown below in JSON format: 

<core_modules> 
{core_modules_description} 
</core_modules> 

These modules are general for all students without any personalization.  
Overall, we want to develop an entrepreneurial mindset via an integrated learning approach, which includes practical elements such as learning logs, projects, case studies, brainstorming, prototyping, testing, personal reflections, self-directed assignments, and ideation exercises. 

# STUDENT # 

The student provided the following background information (Q1-Q18): 

<student_data> 
{student_information} 
</student_data> 

# TASK # 

Create a personalized Smart Learning Plan that deepens the student’s skills. The response must be in Markdown format with the following structure where you need to write parts inside parenthesis [...]: 

------- 
<planning> 
[your detailed internal plan on how to deepen the student’s skills] 
</planning> 

<Smart_Learning_Plan> 
# Smart learning plan (onboarding) 

Hi there! :)

Based on your survey responses, you already have a at least basic understanding of the core topics. This plan provides additional goals, exercises, and resources to help you improve further. 

## 1. Advanced learning goals 

[Taking into account student background and industry, develop 1-3 learning goals for the student to deepen his/her skills and prepare for the training period.] 

## 2. Your tailored study plan 

[Develop step-by step plan for reaching advanced learning goals listed above.] 

## 3. Extra assignments 

[Develop 2-4 small and engaging personalized assignments for the student to test his/her skills. Each assignment needs students to apply generative AI to solve a problem and explain the process with tools and prompts they used.] 

{ending_text} 
</Smart_Learning_Plan> 
------- 

# INSTRUCTIONS # 

Analyze the student’s provided background information (Q1–Q18) to understand his/her skills, industry focus, interests and goals. Consider how the training topic can support the student to reach his/her short and long-term goals. 

Final Output Structure: The final output should be written entirely in MARKDOWN, contained within <Smart_Learning_Plan> section with all planning steps explained in <planning> section. You can only write parts marked inside parenthesis [...], otherwise keep format same. 

Important: 
-Do NOT include timetable for the plan (don't include "Week 1" or "Day 1" or similar). Student studies in his/her own pace. 
-Plan is targeted for learning at home in max 1 week, so do not include complex and long-term tasks/goals, such as "participate in networking events" or "enroll to local University" 
-Do NOT simply copy-paste list of topic as listed above, but adapt them into suitable learning goals for the student   
-You MUST take into account skill levels and industry focus of the student. 
-Remember that aim of this plan is for a student to PREPARE to the actual training of core modules provided by the teacher, not to replace teacher.  

Now, following all above instructions and given plan structure, write the complete personalized Smart Learning Plan for the student.  
Remember to use Markdown format and include the plan inside <Smart_Learning_Plan> tags. 
'''

PROMPT_TEMPLATE_PHASE2_4 = ''' 
# ROLE # 

You are a teacher tasked with creating a personalized Smart Learning Plan (SLP) for a young adult student to support his/her learning and entrepreneurship.

# CONTEXT # 

The training session topics are divided into four (4) core modules. Each module has general (non-personalised) objectives and assignments. This is shown below in JSON format: 

<core_modules> 
{core_modules_description} 
</core_modules> 

These modules are general for all students without any personalization.  
Overall, we want to develop an entrepreneurial mindset via an integrated learning approach, which includes practical elements such as learning logs, projects, case studies, brainstorming, prototyping, testing, personal reflections, self-directed assignments, and ideation exercises. 

# STUDENT # 

The student provided the following background information is him/herself: 

<student_data> 
{student_information} 
</student_data> 

# TASK # 

We want to provide the student a personalized Smart Learning Plan to support general teaching. You must create a personalized Smart Learning Plan for the student to support him/her during the training phase. Your answer must be in Markdown format with the following structure where you must write parts inside parenthesis [...]. 

--------------- 
<internal_planning> 
[your detailed internal thinking and planning how to write the Smart Learning Plan] 
</internal_planning> 

<Smart_Learning_Plan> 
# Smart learning plan (training) 

Hi there! :)

These recommendations are designed to support your preparation for module {module_number}. 

## Learning objectives 

[For module {module_number}, define a clear personalized learning objectives for the student. These objectives must have a clear industry focus that aligns with student background, industry and aims. These objectives should be aligned with the general (non-personalised) objective defined for module {module_number}, but add extra details for each student] 

## Assignments 

[For each learning objective, develop a personalized small and fun assignment to test their skills with industry focus (total 4 assignments for one module). Each assignment needs students to apply generative AI to solve a problem and explain the process with tools and prompts they used. These assignments should be aligned with the general (non-personalised) assignments defined for module {module_number}, but add extra details for each student] 

__We hope these recommendations will help you prepare to module {module_number}.  If you have any questions, please contact teachers.__ 
</Smart_Learning_Plan> 
--------------- 

# INSTRUCTIONS # 

Analyze the student’s provided background information to understand his/her skills, industry focus, interests and goals. Consider how the training topic of module {module_number} can support the student to reach his/her short and long-term goals. 

Final Output Structure: The final output must be written entirely in MARKDOWN, contained within <Smart_Learning_Plan> section with all planning steps explained in <planning> section.  
When writing SLP, use clear structure and bullet-points. 

Important: 
-Use the provided format of the output where you ONLY complete the parts pointed by parenthesis [...]
-This plan is targeted to support topics of MODULE {module_number} described in <core_modules>
-Do NOT include detailed timetable (e.g., specific dates) for the plan. Student studies in his/her own pace. 
-DO NOT simply copy-paste of core topics or assignments, the plan must be adapted for the student 
-Think which topics are most relevant for this particular student taken into account his preferences and business aims

Now, following all above instructions and given plan structure, write the complete personalized, short to long-term Smart Learning Plan for the student.  
Remember to use Markdown format and include the plan inside <Smart_Learning_Plan> tags.
'''

PROMPT_TEMPLATE_ADDITIONAL_MATERIALS = '''
# TASK # 

You are a smart assistant tasked with recommending personalized learning materials for a student.

# STUDENT BACKGROUND INFORMATION #

<student_information>
{student_information}
</student_information>

# STUDENT LEARNING PLAN #

<student_learning_plan>
{learning_plan}
</student_learning_plan>

# LEARNING MATERIALS #

Below is a list of {learning_materials_count} items:

<curated_materials>
{learning_materials}
</curated_materials>

# INSTRUCTIONS #

From the LEARNING MATERIALS list, select 4 OPTIMAL MATERIALS that best suit the student’s needs. Return materials as a Python list of unique numbers.

# OUTPUT FORMAT #

Respond with a Python integer list of length 4. For example, if you pick materials 1, 2, 3 and 4 you would provide a list "[1,2,3,4]".
'''

PROMPT_TEMPLATE_MILESTONES = '''
# ROLE #

You are a teacher tasked with creating a set of learning milestones for a student to support his/her studies and entrepreneurship goals. 

# CONTEXT #

Aim is to teach the student about the following four (4) core modules given below in JSON format:

<core_modules>
{core_modules_description}
</core_modules>

These modules are general for all students without any personalization. 
Overall, we want to develop an entrepreneurial mindset via an integrated learning approach, which includes practical elements such as learning logs, projects, case studies, brainstorming, prototyping, testing, personal reflections, self-directed assignments, and ideation exercises.

# STUDENT BACKGROUND INFORMATION #

Student has provided the following information of him/herself, given inside <student_information> tags:

<student_information>
{student_information}
</student_information>

# SMART LEARNING PLAN #

This is the personalized Smart Learning Plan of the student, given inside <Smart_Learning_Plan> tags:

<Smart_Learning_Plan>
{learning_plan}
</Smart_Learning_Plan>

# TASK #

Your task is to create a list of milestones based on training topics and the learning plan. Based on the Smart Learning Plan, create a list of milestones that student should accomplish during his/her training. These milestones must be related to core modules and the SLP.
Each milestone is one logical and sequential step in the learning process. Milestones must be personalized for the student, supporting his/her particular aims and goals.

# OUTPUT FORMAT #

In you response, provide a list in the following format inside <milestones> tags with short descriptions inside parenthesis [...]:

<milestones>
<milestone1>[short description of the milestone]</milestone1>
<milestone2>[short description of the milestone]</milestone2>
...
<milestoneN>[short description of the milestone]</milestoneN>
</milestones>

Important:
-The number N of milestones depends on the length and content in the learning plan and MUST BE BETWEEN 6-10. Never create more than 10 milestones!
-All milestones must be related to the learning plan and personalized for the student
-Milestones must be practical, clear and logical. Follow a good pedagogical process in creating milestones.
'''

PROMPT_TEMPLATE_ASSISTANT = '''
# ROLE #

You are a smart "UPBEAT Learning Assistant" whose task is to help a student in learning and building his/her skill set. You mentor and support the student in his/her studies and learning in any way you can.

# CONTEXT #

The training session topics are divided into four core modules. Each module has objectives and some pre-defined assingments. This is shown below in JSON format:

{core_modules_description}

Overall, we want to develop an entrepreneurial mindset via an integrated learning approach, which includes practical elements such as learning logs, projects, case studies, brainstorming, prototyping, testing, personal reflections, self-directed assignments, and ideation exercises. 

# STUDENT #

The student has provided the following information of himself/herself via a survey that contains 18 questions [Q1-Q18]:

<student_data>
{student_information}
</student_data>

Questions Q11.1 - Q11.8 are related to student starting skills in the beginning of training period.
Always remember this student information and use it to personalize your responses for the student. 
In particular, take account student skill level and student responses to questions Q16, Q17, Q18 that describe industry focus, aims and motivations.
Student has a personal learning plan (called Smart Learning Plan).

# TASK #

Your task is to help the student to his/her studies and learning. You provide personalized learning assistance and mentoring. 

# INSTRUCTIONS #

-You have access to a personal learning plan (Smart Learning Plan) of the student, which you may discuss about
-Always reply so that the student gets personalized assistance based on his/her background information with potential industry focus
-Always maintain a friendly, supportive tone that encourages self-paced learning and growth of the student

**IMPORTANT:** You can only discuss about things related to learning and studying. If student asks about some completely other topics not related to studying, learning, entrepreneurship, AI or other teaching topics, simply respond "As a learning assistant, I can only discuss about learning topics." 
'''


# -----------------------------
# Utility Functions
# -----------------------------
def read_text_file(filepath):
    """Read text file with error handling"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file {filepath}: {str(e)}")
        raise


def create_modern_happy_smiley():
    """Creates a modern flat-style happy smiley image and returns it as a Base64 data URI."""
    size = (64, 64)
    img = Image.new("RGBA", size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    face_color = "#FFCC00"
    outline_color = "#000000"
    eye_color = "#000000"
    highlight_color = "#FFFFFF"
    draw.ellipse((4, 4, 60, 60), fill=face_color, outline=outline_color, width=3)
    draw.ellipse((18, 22, 26, 30), fill=eye_color)
    draw.ellipse((20, 24, 22, 26), fill=highlight_color)
    draw.ellipse((38, 22, 46, 30), fill=eye_color)
    draw.ellipse((40, 24, 42, 26), fill=highlight_color)
    draw.arc((16, 31, 48, 47), start=0, end=180, fill=outline_color, width=3)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def generate_unique_custom_id(existing_ids, length=15):
    """Generates a unique random ID of specified length."""
    while True:
        ts_hex = f"{time.time_ns():x}"
        rand_hex = f"{random.getrandbits(64):016x}"
        raw_id = (ts_hex + rand_hex)[:length]
        if raw_id not in existing_ids:
            return raw_id


def save_pdf_file(file_path, data):
    """Saves binary PDF data to a file."""
    with open(file_path, 'wb') as pdf_file:
        pdf_file.write(data)


def markdown_to_pdf(md_text):
    """Converts Markdown text to PDF binary data."""
    html_text = markdown2.markdown(md_text)
    full_html = f"""
    <html>
    <head>
      <meta charset="UTF-8">
      <style>
        body {{
          font-family: 'DejaVuSans', Arial, sans-serif;
          font-size: 12pt;
          line-height: 1.6;
        }}
        h1 {{ color: darkblue; text-align: center; }}
        h2 {{ color: darkred; }}
        ul {{ margin-left: 20px; }}
        a {{ color: darkgreen; text-decoration: none; }}
      </style>
    </head>
    <body>{html_text}</body>
    </html>
    """
    pdf_buffer = io.BytesIO()
    pisa_status = pisa.CreatePDF(io.StringIO(full_html), dest=pdf_buffer)
    if pisa_status.err:
        raise Exception("Error in PDF generation")
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()


def pattern_replace(input_text, substring, target_pattern, replacement_text):
    """
    Within each occurrence of 'substring' in input_text, replace the literal target_pattern with replacement_text.
    Returns the modified text and total count of replacements.
    """
    escaped_substring = re.escape(substring)
    escaped_target = re.escape(target_pattern)
    count = 0

    def replacer(match):
        nonlocal count
        original = match.group(0)
        replaced, sub_count = re.subn(escaped_target, replacement_text, original)
        count += sub_count
        return replaced

    output_text = re.sub(escaped_substring, replacer, input_text)
    return output_text, count


def get_llm_response(prompt, llm_config):
    """Calls the LLM API using the provided prompt and configuration."""
    has_temperature = 0 if any([y in llm_config['model'] for y in ['o1', 'o3']]) else 1
    failed_count = 0
    response = None
    while failed_count < 3:
        try:
            print(f"calling LLM {llm_config['model']}...", end='')
            if LOCAL_MODEL:
                response = client.chat.completions.create(
                    model=llm_config['model'],
                    messages=[{'role': "user", 'content': prompt}],
                    temperature=llm_config['temperature'],
                )
            else:
                if has_temperature:
                    response = completion(
                        model=llm_config['model'],
                        temperature=llm_config['temperature'],
                        messages=[{'role': "user", 'content': prompt}],
                        max_tokens=llm_config['max_tokens'],
                    )
                else:
                    response = completion(
                        model=llm_config['model'],
                        messages=[{'role': "user", 'content': prompt}],
                        max_tokens=llm_config['max_tokens'],
                    )
            print("...success")
            break
        except Exception as ex:
            print(f"...FAILED: {ex}")
            failed_count += 1
    if response is None:
        raise Exception("Failed to get LLM response after several attempts.")
    return response.choices[0].message.content


def extract_plan(raw_smart_plan):
    """Extracts the plan content from within <Smart_Learning_Plan> tags."""
    start = raw_smart_plan.find('<Smart_Learning_Plan>')
    if start == -1:
        raise Exception("Missing <Smart_Learning_Plan> tag")
    plan = raw_smart_plan[start + len('<Smart_Learning_Plan>'):]
    end = plan.find('</Smart_Learning_Plan>')
    if end == -1:
        raise Exception("Missing </Smart_Learning_Plan> tag")
    return plan[:end].strip()


def create_plan_prompt(incoming_survey_data, core_modules_data, study_materials, phase=1):
    """
    Constructs the student information prompt and selects the appropriate LLM template based on phase.
    For phase1, if beginner skill gaps are detected, it injects the curated materials from study_materials
    directly into the prompt.
    Returns: (prompt, student_information_prompt, recommended_materials, student_id)
    """
    valid_rows = [k for k in incoming_survey_data.index]
    student_information_prompt = ""
    for key, value in incoming_survey_data.items():
        if key in valid_rows:
            student_information_prompt += f"{key} {value}\n"
    # Identify beginner skill gaps from Q11 questions
    count = 0
    skill_gaps = ""
    recommended_materials_keys = []
    for key, value in incoming_survey_data.items():
        if '[options 1-3]' in key and 'Beginner' in value:
            count += 1
            skill_gaps += f"{key}\n".replace(':','')
            recommended_materials_keys.append(key)
    # If there are beginner gaps, pull the exact curated materials from study_materials
    if count > 0:
        beginner_materials_list = []
        beginner_materials_str = ''
        for k, key in enumerate(recommended_materials_keys):
            material = [y for y in study_materials.index if y in key][0]
            beginner_materials_list.append(str(study_materials.loc[material, 'English']))
            beginner_materials_str += f"\n### {k + 1} {material}  \n{beginner_materials_list[-1]}\n"
        template = PHASE1_PROMPT_TEMPLATE_BEGINNER
    else:
        beginner_materials_str = ""
        template = PHASE1_PROMPT_TEMPLATE_ADVANCED

    #template = template.replace('{student_name}', incoming_survey_data['Q1. Full Name'].strip())

    if phase == 1:
        prompt = template.replace('{student_information}', student_information_prompt)
        prompt = prompt.replace('{core_modules_description}', core_modules_data)
        prompt = prompt.replace('{total_assignment_count}', str(count * 2))
        prompt = prompt.replace('{skill_gaps}', skill_gaps)
        prompt = prompt.replace('{beginner_level_materials}', beginner_materials_str)
        prompt = prompt.replace('{skill_gaps_count}', str(count))
        prompt = prompt.replace('{ending_text}', ending_text)
    elif phase > 1:
        prompt = PROMPT_TEMPLATE_PHASE2_4.replace('{student_information}', student_information_prompt)
        prompt = prompt.replace('{core_modules_description}', core_modules_data)
        prompt = prompt.replace('{module_number}', str(phase))
    else:
        raise ValueError("Invalid phase specified.")
    # Retrieve student identifier from a key field (for example, Q1. Full Name)
    return prompt, student_information_prompt, recommended_materials_keys


# -----------------------------
# SmartPlanGenerator Class
# -----------------------------
class SmartPlanGenerator:
    def __init__(self, core_modules_data, study_materials, additional_courses_data, llm_config_large, llm_config_small,
                 happy_img):
        self.core_modules_data = core_modules_data
        self.study_materials = study_materials
        self.additional_courses_data = additional_courses_data
        self.llm_config_large = llm_config_large
        self.llm_config_small = llm_config_small
        self.happy_img = happy_img

    def remove_border_parentheses(self, text):
        return re.sub(r'^[\[\(\{](.*?)[\]\)\}]$', r'\1', text)

    def generate_milestones(self, plan, student_data, core_modules_data):

        valid_rows = [k for k in student_data.index if
                      any(x in k for x in
                          ['Q1.', 'Q8.', 'Q9.', 'Q10.', 'Q15.', 'Q16.', 'Q17.', 'Q18.', 'Additional Information'])]
        student_information_prompt = ""
        for key, value in student_data.items():
            if key in valid_rows:
                student_information_prompt += f"{key}: {value}\n"
        prompt = PROMPT_TEMPLATE_MILESTONES.replace('{learning_plan}', plan)
        prompt = prompt.replace('{student_information}', student_information_prompt)
        prompt = prompt.replace('{core_modules_description}', core_modules_data)
        milestones_raw = get_llm_response(prompt, self.llm_config_large)
        milestones = [self.remove_border_parentheses(x).strip() for x in
                      re.findall(r'<milestone\d+>\s*(.*?)\s*</milestone\d+>', milestones_raw)]
        return milestones_raw, milestones

    def generate_phase_plan(self, student_data, phase):
        """
        Generates the smart plan for a given phase.
        Returns a dictionary with plan_prompt, smart_plan, pdf content, student_info, recommended_materials, and student_id.
        """
        prompt, student_info, recommended_materials = create_plan_prompt(student_data,self.core_modules_data,self.study_materials, phase=phase)

        if '# TASK #' in prompt:  # LLM is needed
            raw_plan = get_llm_response(prompt, self.llm_config_large)
            plan_content = extract_plan(raw_plan)
        else:
            raw_plan = prompt
            plan_content = prompt

        # if phase == 1:
        # plan_content = plan_content.replace(ending_text,'')

        return {
            'plan_prompt': prompt,
            'smart_plan': plan_content,
            'student_info': student_info,
            'recommended_materials': recommended_materials,
        }

    def append_additional_materials(self, plan_content, student_info, recommended_materials):
        """
        Appends additional online materials to the plan using the curated list.
        """
        prompt = PROMPT_TEMPLATE_ADDITIONAL_MATERIALS.replace('{student_information}', student_info)
        prompt = prompt.replace('{learning_plan}', plan_content)
        learning_materials_prompt = ""
        for idx, row in self.additional_courses_data.iterrows():
            learning_materials_prompt += f"{idx}: {row['description']}\n"
        prompt = prompt.replace('{learning_materials}', learning_materials_prompt)
        prompt = prompt.replace('{learning_materials_count}', str(self.additional_courses_data.shape[0]))
        materials_response = get_llm_response(prompt, self.llm_config_small)
        extracted_list_match = re.search(r'\[.*?\]', materials_response)
        if extracted_list_match is None:
            raise Exception("Failed to extract materials list.")
        extracted_list = ast.literal_eval(extracted_list_match.group(0))
        extracted_list = np.unique([int(x) for x in extracted_list])
        if min(extracted_list) < 1 or max(extracted_list) > self.additional_courses_data.shape[0]:
            raise Exception("Extracted materials list is out of bounds.")
        df = self.additional_courses_data.loc[extracted_list]
        additional_materials_str = ""
        for k, (index, row) in enumerate(df.iterrows(), start=1):
            additional_materials_str += f"{k}: {row['description']}<br>{row['url']}\n\n"
        section_num = 5 if len(recommended_materials) > 0 else 4
        additional_section = f"## {section_num}. Additional online materials\n\n{additional_materials_str}\n"
        updated_plan = plan_content + "\n" + additional_section
        # Optionally, replace a smiley placeholder if found.

        target_substring = " :) "
        target_pattern = ":)"
        ending_text_smiley, replace_count = pattern_replace(ending_text, " :) ", ":)",
                                                            f'<img src="{self.happy_img}" alt="happy" width="20" height="20">')
        assert replace_count == 1
        updated_plan_to_pdf = updated_plan + ending_text_smiley
        updated_plan += ending_text

        return updated_plan, updated_plan_to_pdf

# -----------------------------
# Main Processing Function
# -----------------------------
def main():
    # Load student survey, study materials and curated materials data
    additional_courses_data = pd.read_csv(r'data/curated_additional_materials.txt', sep='|', index_col=0)
    core_modules_data = read_text_file(r'data/description_of_training.txt')
    study_materials = pd.read_excel(r'data/beginner_materials.xlsx', index_col=0)

    incoming_survey_data = pd.read_excel(r'data/upbeat_LA_data_combined_9.4.2025.xlsx',header=[0],skiprows=[0]).transpose()
    selected = incoming_survey_data.iloc[:,0].apply(lambda x:x==1)
    incoming_survey_data = incoming_survey_data.loc[selected,:]
    incoming_survey_data = incoming_survey_data.iloc[:,1:]
    incoming_survey_data.fillna('NaN', inplace=True)
    incoming_survey_data = incoming_survey_data.map(lambda x: x.replace('\n', ' ').strip())

    skill_strings = {'Продвинутый: 3':'Advanced [3]','Базовый: 2':'Basic [2]','Новичок: 1':'Beginner [1]',
                     'Advanced: 3':'Advanced [3]','Basic: 2':'Basic [2]','Beginner: 1':'Beginner [1]'}
    for k,v in skill_strings.items():
        mask = incoming_survey_data.map(lambda x: k in x)
        incoming_survey_data[mask] = v

    incoming_survey_data.index = [
        next((f"{y} skill-level [options 1-3]:" for y in study_materials.index if y in x), x)
        for x in incoming_survey_data.index
    ]
    incoming_survey_data.index = [x.strip() for x in incoming_survey_data.index]

    anonymous_animals = [
        "Alligator", "Anteater", "Armadillo", "Aurochs",
        "Axolotl", "Badger","Beaver","Unicorn",
        "Blobfish", "Buffalo", "Camel", "Capybara",
        "Chameleon", "Cheetah", "Chinchilla", "Chipmunk",
        "Chupacabra", "Coyote", "Crow",
        "Dingo", "Dinosaur", "Dog", "Dolphin",
        "Dragon","Elephant",
        "Ferret", "Frog", "Giraffe",
        "Gopher", "Grizzly", "Hedgehog",
        "Hyena", "Ibex", "Ifrit", "Iguana",
        "Jackal", "Kangaroo", "Koala",
        "Kraken", "Leopard", "Lion", "Liger",
        "Lemur", "Llama", "Mink",
        "Moose", "Narwhal",
        "Otter", "Panda", "Penguin",
        "Platypus", "Pumpkin", "Python", "Quagga",
        "Quokka", "Rabbit", "Raccoon", "Rhino",
        "Shrew", "Squirrel", "Tiger", "Turtle", "Unicorn",
        "Wolf", "Wolverine", "Wombat"
    ]
    anonymous_animals = list(set(anonymous_animals))

    assert incoming_survey_data.shape[1] <= len(anonymous_animals),'not enough animals, add more animal labels in list!'

    plan_output_path = r'learning_plans'

    # Create a happy smiley image for later use
    happy_img = create_modern_happy_smiley()

    # Initialize the plan generator with study_materials included
    plan_generator = SmartPlanGenerator(core_modules_data, study_materials, additional_courses_data, llm_config_large,
                                        llm_config_small, happy_img)

    study_plans = {}
    existing_passwords = set()
    # Iterate over each student (each column represents one student's survey responses)
    for plan_k, col in enumerate(incoming_survey_data.columns):
        print(f'generating plan {plan_k + 1} of {incoming_survey_data.shape[1]}')
        student_data = incoming_survey_data[col]

        student_username = anonymous_animals[plan_k]

        password = generate_unique_custom_id(existing_passwords, length=10)
        existing_passwords.add(password)

        # Generate plan for phase 1 (onboarding)
        # phase1_result = plan_generator.generate_phase_plan(student_data, phase=1)
        # Append additional online materials to phase 1 plan

        # updated_plan_phase1,updated_plan_phase1_to_pdf = plan_generator.append_additional_materials(
        #    phase1_result['smart_plan'], phase1_result['student_info'], phase1_result['recommended_materials']
        # )
        # pdf_content_phase1 = markdown_to_pdf(updated_plan_phase1_to_pdf)

        # Generate plan for phase 2 (training)
        phase1_result = plan_generator.generate_phase_plan(student_data, phase=1)
        phase2_result = plan_generator.generate_phase_plan(student_data, phase=2)
        phase3_result = plan_generator.generate_phase_plan(student_data, phase=3)
        phase4_result = plan_generator.generate_phase_plan(student_data, phase=4)

        pdf_content_phase1 = markdown_to_pdf(phase1_result['smart_plan'])
        pdf_content_phase2 = markdown_to_pdf(phase2_result['smart_plan'])
        pdf_content_phase3 = markdown_to_pdf(phase3_result['smart_plan'])
        pdf_content_phase4 = markdown_to_pdf(phase4_result['smart_plan'])

        milestones_raw, milestones = plan_generator.generate_milestones(phase4_result['smart_plan'],student_data,core_modules_data)

        # Generate assistant prompt for chatbot usage
        assistant_prompt = PROMPT_TEMPLATE_ASSISTANT.replace('{student_information}', phase1_result['student_info'])
        assistant_prompt = assistant_prompt.replace('{core_modules_description}', core_modules_data)
        # Generate a unique password for the student

        # Store the study plans for this student
        study_plans[student_username] = {
            'plan_prompt_phase1': phase1_result['plan_prompt'],
            'plan_prompt_phase2': phase2_result['plan_prompt'],
            'plan_prompt_phase3': phase3_result['plan_prompt'],
            'plan_prompt_phase4': phase4_result['plan_prompt'],
            'smart_plan_phase1': phase1_result['smart_plan'],
            'smart_plan_phase2': phase2_result['smart_plan'],
            'smart_plan_phase3': phase3_result['smart_plan'],
            'smart_plan_phase4': phase4_result['smart_plan'],
            'smart_plan_pdf_phase1': pdf_content_phase1,
            'smart_plan_pdf_phase2': pdf_content_phase2,
            'smart_plan_pdf_phase3': pdf_content_phase3,
            'smart_plan_pdf_phase4': pdf_content_phase4,
            'milestones': milestones,
            'data': student_data,
            'assistant_prompt': assistant_prompt,
            'password': password,
            'username':student_username
        }
        print(f"Study plans for student '{student_username}' created.")

        # Save the study plans to a pickle file and CSV
        with open(plan_output_path + os.sep + 'study_plans_data.pickle', 'wb') as f:
            pickle.dump(study_plans, f)

        save_pdf_file(plan_output_path + os.sep + f'{student_username}_plan_phase1.pdf',
                      study_plans[student_username]['smart_plan_pdf_phase1'])
        save_pdf_file(plan_output_path + os.sep + f'{student_username}_plan_phase2.pdf',
                      study_plans[student_username]['smart_plan_pdf_phase2'])
        save_pdf_file(plan_output_path + os.sep + f'{student_username}_plan_phase3.pdf',
                      study_plans[student_username]['smart_plan_pdf_phase3'])
        save_pdf_file(plan_output_path + os.sep + f'{student_username}_plan_phase4.pdf',
                      study_plans[student_username]['smart_plan_pdf_phase4'])

    # print all plans as CSV
    df_rows = []
    for sid, details in study_plans.items():
        df_rows.append({
            'username': sid,
            'plan_prompt_phase1': details['plan_prompt_phase1'],
            'plan_prompt_phase2': details['plan_prompt_phase2'],
            'plan_prompt_phase3': details['plan_prompt_phase3'],
            'plan_prompt_phase4': details['plan_prompt_phase4'],
            'smart_plan_phase1': details['smart_plan_phase1'],
            'smart_plan_phase2': details['smart_plan_phase2'],
            'smart_plan_phase3': details['smart_plan_phase3'],
            'smart_plan_phase4': details['smart_plan_phase4'],
            'assistant_prompt': details['assistant_prompt'],
            'milestones': details['milestones'],
            'password': details['password']
        })

    df_output = pd.DataFrame(df_rows)
    df_output.to_csv(plan_output_path + os.sep + 'study_plans_data.csv', index=False)

    print("All study plans generated and saved.")

if __name__ == '__main__':
    main()
