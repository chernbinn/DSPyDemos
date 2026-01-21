# Optional
#os.environ["FAL_API_KEY"] = "your_fal_api_key"
#os.environ["OPENAI_API_KEY"] = "your_openai_api_key"

import dspy

from PIL import Image
from io import BytesIO
import requests
import fal_client

from dotenv import load_dotenv
load_dotenv()

# import display
from IPython.display import display

lm = dspy.LM(model="ollama/llama3.2-vision:11b", temperature=0.5)
dspy.configure(lm=lm)

def generate_image(prompt):

    request_id = fal_client.submit(
        "fal-ai/flux-pro/v1.1-ultra",
        arguments={
            "prompt": prompt
        },
    ).request_id

    result = fal_client.result("fal-ai/flux-pro/v1.1-ultra", request_id)
    url = result["images"][0]["url"]

    return dspy.Image.from_url(url)

def display_image(image):
    url = image.url
    # download the image
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))

    # display at 25% of original size
    display(image.resize((image.width // 4, image.height // 4)))

check_and_revise_prompt = dspy.Predict("desired_prompt: str, current_image: dspy.Image, current_prompt:str -> feedback:str, image_strictly_matches_desired_prompt: bool, revised_prompt: str")

initial_prompt = "A scene that's both peaceful and tense"
current_prompt = initial_prompt

max_iter = 5
for i in range(max_iter):
    print(f"Iteration {i+1} of {max_iter}")
    current_image = generate_image(current_prompt)
    result = check_and_revise_prompt(desired_prompt=initial_prompt, current_image=current_image, current_prompt=current_prompt)
    display_image(current_image)
    if result.image_strictly_matches_desired_prompt:
        break
    else:
        current_prompt = result.revised_prompt
        print(f"Feedback: {result.feedback}")
        print(f"Revised prompt: {result.revised_prompt}")

print(f"Final prompt: {current_prompt}")

dspy.inspect_history(5)