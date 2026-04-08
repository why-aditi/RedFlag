
import sys
import os
from unittest.mock import MagicMock

# Add current dir to path
sys.path.append(os.getcwd())

from models import RedflagAction
from server.redflag_env_environment import RedflagEnvironment

def test_step():
    env = RedflagEnvironment()
    # Mock task registry to avoid loading real tasks
    env._task_ids = ["test"]
    env.reset = MagicMock(return_value=MagicMock())
    
    # We want to see if step returns a tuple
    action = RedflagAction(action_type="approve")
    
    # Manually setup episode mock
    env._episode = MagicMock()
    env._episode.step = 0
    env._episode.max_steps = 8
    env._episode.done = False
    env._episode.step_rewards = []
    
    # Mock build_observation
    mock_obs = MagicMock()
    mock_obs.done = True
    mock_obs.metadata = {"test": "info"}
    env._build_observation = MagicMock(return_value=mock_obs)
    
    result = env.step(action)
    print(f"Result type: {type(result)}")
    if isinstance(result, tuple):
        print(f"Result length: {len(result)}")
        obs, reward, done, info = result
        print("Success: Step returns 4-tuple")
    else:
        print("Failure: Step did not return 4-tuple")

if __name__ == "__main__":
    test_step()
