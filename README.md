# nlp-project


2-1 split: you can ask each other "why this, why not that"

## Internal: 
#### Stage 1: 
March 10th deadline (midterms) **1 Viraj + 2 (Bharadhwaj + Abhishek)** 
- everyone learns LangGraph API. Sets up WSL
- 2 people setup docker/devcontainers for all to use + designing benchmark task to test for + set up testing pipeline. Test performance of 2-agent system.
	- Viraj: Need testing/accuracy infrastructure to evaluate agent outcomes
- 1 person set up a working 2-agent system [Thinking in LangGraph - Docs by LangChain](https://docs.langchain.com/oss/python/langgraph/thinking-in-langgraph) 
	- relatively less work
#### Stage 2: 
March 25th deadline: **1 Bharadwaj + 2 (Viraj + Abhishek)**
- 2 people set up fully functional pipeline 
	- Abhishek: human-in-the-loop AND wandb experiment tracking
	- Abhishek: Graph Architecture - notes edges of agent communication and enabling modularity so we can make changes
	- Abhishek: capability extension: vectorDb (chromadb) for RAG
- 1 person research on modelling agent interactions - HAS to work on designing exp on stage 3
	- Option: Look into network topology (for collaboration graph) and/or changing roles of each agent
#### Stage 3: 
April 15th deadline **1 Abhishek + 2 (Viraj + Bharadhwaj)**
- 2 people coding the above models/experiments 
	- Viraj: Hard coding the scoping of planner agent's capacities
- 1 person running them and collecting data
		- relatively less hands-on work
## External: 
- Mid Stage Presentation - 4.20
- Final Presentation 5.5
- Final Report - 5.14 - Viraj and Abhishek


## Implementation-level topics:
- Ask Abhishek for: We include an edge case or critic agent that develops its own test cases outside the purview of the pipeline. Code coverage is important but no meaningless bloat 
