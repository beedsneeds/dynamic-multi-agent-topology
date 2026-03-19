import os
from dotenv import load_dotenv
from langsmith import Client

# Load environment variables from .env file
load_dotenv()

def init_langsmith(project_name: str = "multi-agent-baseline"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = project_name
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")

    client = Client()

    print(f"[LangSmith] Tracing enabled - project: '{project_name}'")

    return client


class Projects:
    BASELINE    = "multi-agent-baseline"
    EXPERIMENTS = "multi-agent-experiments"
    ABLATIONS   = "multi-agent-ablations"


if __name__ == "__main__":
    client = init_langsmith(Projects.BASELINE)

    projects = list(client.list_projects())
    print(f"\n[LangSmith] Connected. Found {len(projects)} existing project(s).")
    for p in projects:
        print(f"  - {p.name}")