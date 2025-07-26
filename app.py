from smolagents import CodeAgent,DuckDuckGoSearchTool,load_tool,tool ,HfApiModel
import datetime
import requests
import pytz
import yaml
from tools.final_answer import FinalAnswerTool

from Gradio_UI import GradioUI

# Below is an example of a tool that does nothing. Amaze us with your creativity !
@tool
def artistInfo(artist:str)-> str:
    """A tool that fetches information about an artist.
    Args:
        artist: A string representing the name of the artist.
    """
    try:
        DuckDuckGoSearchTool.setup()
        web_search_tool = DuckDuckGoSearchTool(max_results=5, rate_limit=2.0)
        # FIX 1: Use an f-string to insert the artist's name
        results = web_search_tool(f"{artist} artist information")
        search_results = [result['title'] + ": " + result['body'] for result in results if 'title' in result and 'body' in result]
        if not search_results:
            return f"No information found for artist '{artist}'."
        return f"Information about {artist}: {search_results[0]}"
    except Exception as e:
        return f"Error fetching information for artist '{artist}': {str(e)}"
       
    

@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """A tool that fetches the current local time in a specified timezone.
    Args:
        timezone: A string representing a valid timezone (e.g., 'America/New_York').
    """
    try:
        # Create timezone object
        tz = pytz.timezone(timezone)
        # Get current time in that timezone
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"


final_answer = FinalAnswerTool()

# If the agent does not answer, the model is overloaded, please use another model or the following Hugging Face Endpoint that also contains qwen2.5 coder:
# model_id='https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud' 

model = HfApiModel(
max_tokens=2096,
temperature=0.5,
model_id='Qwen/Qwen2.5-Coder-32B-Instruct',# it is possible that this model may be overloaded
custom_role_conversions=None,
)


# Import tool from Hub
image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True)

with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)
    
agent = CodeAgent(
    model=model,
    tools=[final_answer, artistInfo, get_current_time_in_timezone],  ## add your tools here (don't remove final answer)
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates
)


    GradioUI(agent).launch()
