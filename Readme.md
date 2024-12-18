![AgentExample.png](readme_resources%2FAgentExample.png)

## Prerequisites

Before you begin, ensure you have the following:

- Docker
- Python 3.x
- OpenAI API key

## Setup Instructions

### 1. Add Your OpenAI API Key

A `.env` file already exists in the root directory of the project with the following content:

```plaintext
OPENAI_API_KEY=<your-api-key>
```

Replace `<your-api-key>` with your actual OpenAI API key. This key will be used by the application to interact with the OpenAI API.

### 2. Start the Docker Containers

To start the application, run the following command from the root directory:

```bash
docker compose up
```

This will start all the required Docker containers for the project.

### 3. Configure the Agent in OpenWebUI

The Agent does not support streaming out of the box, but OpenWebUI has Streaming set as a default. To configure the agent accordingly:

1. Open the OpenWebUI interface in your browser at http://localhost:2000/.
2. Follow the steps demonstrated in the video below to correctly configure the Agent, so that it Streaming is turned off and it therefore properly works in OpenWebUI:

![tutorial.gif](readme_resources%2Ftutorial.gif)

## Customizing the Agent's Tools

The `tools.py` file allows you to customize the tools the agent has access to. You can add new tools simply by creating a Python function with the appropriate annotations.

### Example:
```python
def get_weather(
    city: Annotated[str, "The city for which to get the weather"] = "Munich",
    unit: Annotated[Literal["Celsius", "Fahrenheit"], "The unit for the temperature"] = "Celsius"
):
    """
    A simple function to get the current weather for a location.
    """
    return f"22.0 {unit}"
```

Adding a new tool like this enables the agent to perform additional tasks. You can refer to the other example methods already included in `tools.py` for further guidance.

## Haystack Pipeline
The core logic of the Haystack pipeline, powering the agent, is implemented in `agent.py`. Below is a visualization of the pipeline:

![pipeline.png](readme_resources%2Fpipeline.png)
