from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END

# 1. Definimos la 'Memoria' del agente 
class AgentState(TypedDict):
    perfil_paciente: str
    entidades_medicas: list
    ensayos_encontrados: list
    ensayos_validados: list

# 2. Definición de los Nodos (

def nodo_extractor(state: AgentState):
    print("--- EJECUTANDO NODO 1: EXTRACTOR ---")
    # Aquí irá la lógica para limpiar el texto del paciente
    return {"entidades_medicas": []}

def nodo_retriever(state: AgentState):
    print("--- EJECUTANDO NODO 2: BUSCADOR ---")
    # Aquí irá la conexión con ClinicalTrials.gov
    return {"ensayos_encontrados": []}

def nodo_validador(state: AgentState):
    print("--- EJECUTANDO NODO 3: VALIDADOR ---")
    # Aquí Gemini decidirá si el paciente es apto o no
    return {"ensayos_validados": []}

# 3. Construcción del Grafo 
workflow = StateGraph(AgentState)

workflow.add_node("extractor", nodo_extractor)
workflow.add_node("retriever", nodo_retriever)
workflow.add_node("validador", nodo_validador)

workflow.set_entry_point("extractor")
workflow.add_edge("extractor", "retriever")
workflow.add_edge("retriever", "validador")
workflow.add_edge("validador", END)

# Esta es la variable que tus compañeros usarán para ejecutar el proyecto
app = workflow.compile()
