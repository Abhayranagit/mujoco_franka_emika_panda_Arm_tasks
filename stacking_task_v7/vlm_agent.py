# vlm_agent.py
import base64
import requests # Or use the openai library

class VLMAgent:
    def __init__(self, api_key):
        self.api_key = api_key

    def get_next_action(self, image_path, object_states):
        """
        Sends the image and state to the VLM and returns a command string.
        Expected outputs: "pick(cube_3)", "place(cube_1)", or "done"
        """
        # Encode your saved image to base64
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

        prompt = f"""
        You are controlling a Franka Emika Panda arm. 
        Current object states: {object_states}
        Goal: Stack cube_3 on top of cube_1.
        You can only output ONE of these commands: 
        pick(cube_name), place(cube_name), or done.
        Look at the image and output only the exact command.
        """
        
        # --- API CALL LOGIC GOES HERE ---
        # response = requests.post(...)
        
        # For testing, mock the response:
        return "pick(cube_1)" # Example output