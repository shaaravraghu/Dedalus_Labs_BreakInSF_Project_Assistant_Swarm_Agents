# Dedalus_Labs_BreakInSF_Project_Assistant_Swarm_Agents

This is a backend-only project built using **Python** and the **Dedalus Labs SDK**.  
It simulates a swarm of AI agents that collaborate to help engineers plan and develop physical technology projects like RC planes, self-driving cars, high-altitude balloons, or Tesla coils.

## Overview
The system runs entirely in one Python file. It uses multiple agents that call each other to share constraints and micro-solutions:
- **Classifier Agent:** Takes user input and decides which sub-agents are needed.  
- **Sub-Agents:** Each focuses on a specific domain (Technical, Design, Team, Business, Legal, Deployment).  
- **Synchroniser Agent:** Merges all agent outputs, resolves conflicts, and produces the final plan.

## Setup
Install requirements:
```bash
pip install dedalus-labs python-dotenv
