import base64
import os
import sys

from a2a.server.tasks import TaskUpdater
from a2a.types import DataPart, FilePart, FileWithBytes, Message, TaskState, Part, TextPart
from a2a.utils import new_agent_text_message

from messenger import Messenger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../osworld"))
from mm_agents.agent import PromptAgent


class Agent:
    def __init__(self):
        self.messenger = Messenger()
        # Initialize other state here
        self.prompt_agent = PromptAgent(
            platform=os.environ.get("PLATFORM", "ubuntu"),
            model=os.environ.get("MODEL", "gpt-4o"),
            max_tokens=int(os.environ.get("MAX_TOKENS", "1500")),
            top_p=float(os.environ.get("TOP_P", "0.9")),
            temperature=float(os.environ.get("TEMPERATURE", "0.5")),
            action_space=os.environ.get("ACTION_SPACE", "pyautogui"),
            observation_type=os.environ.get("OBSERVATION_TYPE", "screenshot"),
            max_trajectory_length=int(os.environ.get("MAX_TRAJECTORY_LENGTH", "3")),
            a11y_tree_max_tokens=int(os.environ.get("A11Y_TREE_MAX_TOKENS", "10000")),
            client_password=os.environ.get("CLIENT_PASSWORD", "password"),
        )

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """Implement your agent logic here.

        Args:
            message: The incoming message
            updater: Report progress (update_status) and results (add_artifact)

        Use self.messenger.talk_to_agent(message, url) to call other agents.
        """
        # Replace this example code with your agent logic

        await updater.update_status(
            TaskState.working, new_agent_text_message("Thinking...")
        )
        # Unpack parts sent by A2AClientAgent
        instruction = ""
        obs: dict = {}

        for part in message.parts:
            root = part.root
            if isinstance(root, TextPart):
                instruction = root.text
            elif isinstance(root, FilePart):
                if isinstance(root.file, FileWithBytes):
                    obs["screenshot"] = base64.b64decode(root.file.bytes)
            elif isinstance(root, DataPart):
                obs.update(root.data)

        response, actions = self.prompt_agent.predict(instruction, obs)

        await updater.add_artifact(
            parts=[
                Part(root=TextPart(text=response or "")),
                Part(root=DataPart(data={"actions": actions or []})),
            ],
            name="prediction",
        )
