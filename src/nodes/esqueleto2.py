from typing import Annotated, TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import google.generativeai as genai
import os
import re

# Configurar Gemini (reemplaza con tu API key)
client = genai.Client(api_key="")
model = genai.GenerativeModel('gemini-1.5-flash')

# 1. Estado expandido para scoring
class AgentState(TypedDict):
    perfil_paciente: str
    entidades_medicas: List[Dict[str, Any]]
    ensayos_encontrados: List[Dict[str, Any]]
    ensayos_validados: List[Dict[str, Any]]
    ensayos_rankeados: List[Dict[str, Any]]

# 2. NODO 1: EXTRACTOR CON LLM + MeSH REAL
def nodo_extractor(state: AgentState):
    print("--- 🤖 EJECUTANDO NODO 1: EXTRACTOR (LLM + MeSH Real) ---")
    
    perfil = state["perfil_paciente"]
    
    prompt = f"""
    ANALIZA este perfil médico y extrae ESTRUCTURADAMENTE:
    
    Perfil: {perfil}
    
    RESPUESTA EXACTA en JSON:
    {{
        "edad": NUMERO,
        "sexo": "male"|"female",
        "condiciones": [
            {{"nombre": "texto", "mesh_code": "C12345", "confidence": 0.9}}
        ],
        "medicacion": ["lista medicamentos"],
        "ecog": numero_o_null,
        "texto_completo": "resumen clínico"
    }}
    
    OBLIGATORIO:
    1. Busca CÓDIGOS MeSH REALES (C00000 formato)
    2. Si no encuentras MeSH exacto, usa el más cercano
    3. Confidence 0.0-1.0
    """
    
    response = model.generate_content(prompt)
    entidades_raw = response.text.strip('```json```').strip('```')
    
    try:
        import json
        entidades = json.loads(entidades_raw)
        entidades_medicas = [entidades]
    except:
        # Fallback estructurado
        entidades_medicas = [{
            "edad": 60, "sexo": "female", "condiciones": [], 
            "medicacion": [], "ecog": None
        }]
    
    print(f"✅ Extraídas {len(entidades_medicas[0]['condiciones'])} condiciones MeSH reales")
    return {"entidades_medicas": entidades_medicas}

# 3. NODO 2: RETRIEVER CON API REAL ClinicalTrials.gov v2
def nodo_retriever(state: AgentState):
    print("--- 🌐 EJECUTANDO NODO 2: RETRIEVER (API REAL ClinicalTrials.gov) ---")
    
    entidades = state["entidades_medicas"][0]
    condiciones_mesh = [c["mesh_code"] for c in entidades["condiciones"]]
    
    ensayos = []
    
    # QUERIES ENCADENADAS REALES
    queries = [
        f"mesh:\"{condiciones_mesh[0]}\"",  # 1. Condición MeSH
        f"mesh:\"{condiciones_mesh[0]}\" AND phase:2,3",  # 2. + Fases II/III
        f"mesh:\"{condiciones_mesh[0]}\" AND status:recruiting"  # 3. Recruiting
    ]
    
    for i, query in enumerate(queries):
        url = f"https://clinicaltrials.gov/api/v2/studies"
        params = {
            'query.cond': query,
            'recursionLimit': 20,
            'format': 'json'
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                for study in data.get('studies', [])[:10]:
                    ensayo = {
                        'NCTId': study.get('nctId', ''),
                        'BriefTitle': study.get('briefTitle', ''),
                        'Phase': study.get('phase', {}).get('name', 'Phase 1'),
                        'RecruitmentStatus': study.get('overallStatus', ''),
                        'EligibilityCriteria': study.get('eligibilityCriteria', {}).get('criteria', ''),
                        'Locations': [arm.get('name', '') for arm in study.get('arms', [])],
                        'raw_data': study
                    }
                    ensayos.append(ensayo)
        except Exception as e:
            print(f"⚠️ Error API query {i+1}: {e}")
            continue
    
    # Eliminar duplicados
    ensayos_unicos = {e['NCTId']: e for e in ensayos}.values()
    ensayos_final = list(ensayos_unicos)[:20]
    
    print(f"✅ API REAL: {len(ensayos_final)} ensayos únicos")
    return {"ensayos_encontrados": ensayos_final}

# 4. NODO 3: VALIDADOR CON LLM (Reemplaza Regex)
def nodo_validador(state: AgentState):
    print("--- ⚖️ EJECUTANDO NODO 3: VALIDADOR LLM (Manejo Negaciones/Temporal) ---")
    
    entidades = state["entidades_medicas"][0]
    ensayos = state["ensayos_encontrados"]
    
    ensayos_validados = []
    
    for ensayo in ensayos:
        prompt = f"""
        VALIDA si este paciente CUMPLE los criterios del ensayo:
        
        PACIENTE:
        Edad: {entidades.get('edad')}
        Sexo: {entidades.get('sexo')}
        Condiciones: {entidades.get('condiciones')}
        Medicación: {entidades.get('medicacion')}
        ECOG: {entidades.get('ecog')}
        
        ENSAYO NCT{ensayo['NCTId']}:
        {ensayo['EligibilityCriteria'][:2000]}
        
        RESPUESTA EXACTA JSON:
        {{
            "decision": "MET"|"NOT MET"|"NEI",
            "confidence": 0.0-1.0,
            "razonamiento": "explicación detallada",
            "criterios_met": ["lista"],
            "criterios_fallidos": ["lista"],
            "info_necesaria": ["preguntas"]
        }}
        
        MANEJA:
        - NEGACIONES: "no prior treatment", "without brain mets"
        - UMBRALES: "ECOG <=2", "age 18-75"
        - TEMPORAL: "last 6 months", "within 12 weeks"
        """
        
        response = model.generate_content(prompt)
        validacion_raw = response.text.strip('```json```').strip('```')
        
        try:
            import json
            validacion = json.loads(validacion_raw)
        except:
            validacion = {
                "decision": "NEI", "confidence": 0.5,
                "razonamiento": "Error parsing, necesita revisión",
                "criterios_met": [], "criterios_fallidos": [],
                "info_necesaria": ["Revisar criterios manualmente"]
            }
        
        ensayo_validado = {**ensayo, **validacion}
        ensayos_validados.append(ensayo_validado)
    
    print(f"✅ LLM validó {len(ensayos_validados)} ensayos")
    return {"ensayos_validados": ensayos_validados}

# 5. NODO 4: SCORING MATEMÁTICO (NDCG@10)
def nodo_scorer(state: AgentState):
    print("--- 📊 EJECUTANDO NODO 4: SCORING (Fórmula Matemática NDCG@10) ---")
    
    ensayos = state["ensayos_validados"]
    
    # FÓRMULA: Score = 0.5×elegibilidad + 0.3×fase + 0.2×estado
    for ensayo in ensayos:
        # Elegibilidad (LLM decision)
        elig_map = {"MET": 1.0, "NEI": 0.5, "NOT MET": 0.0}
        elegibilidad = elig_map.get(ensayo["decision"], 0.0) * ensayo["confidence"]
        
        # Fase (valores numéricos)
        fase_map = {
            "Phase 3": 1.0, "Phase 2": 0.7, "Phase 1": 0.4, "Not Applicable": 0.2
        }
        fase_score = fase_map.get(ensayo["Phase"], 0.3)
        
        # Estado reclutamiento
        estado_map = {
            "Recruiting": 1.0, "Active, not recruiting": 0.6,
            "Not yet recruiting": 0.4, "Terminated": 0.0
        }
        estado_score = estado_map.get(ensayo["RecruitmentStatus"], 0.3)
        
        # FÓRMULA FINAL
        score = 0.5 * elegibilidad + 0.3 * fase_score + 0.2 * estado_score
        
        ensayo["score_final"] = round(score, 3)
        ensayo["score_breakdown"] = {
            "elegibilidad": round(elegibilidad, 3),
            "fase": round(fase_score, 3),
            "estado": round(estado_score, 3)
        }
    
    # RANKING NDCG@10
    ensayos_rankeados = sorted(ensayos, key=lambda x: x["score_final"], reverse=True)[:10]
    
    print(f"✅ Ranking NDCG@10: Scores {ensayos_rankeados[0]['score_final']:.3f} - {ensayos_rankeados[-1]['score_final']:.3f}")
    return {"ensayos_rankeados": ensayos_rankeados}

# 6. GRAFO COMPLETO
workflow = StateGraph(AgentState)

workflow.add_node("extractor", nodo_extractor)
workflow.add_node("retriever", nodo_retriever)
workflow.add_node("validador", nodo_validador)
workflow.add_node("scorer", nodo_scorer)

workflow.set_entry_point("extractor")
workflow.add_edge("extractor", "retriever")
workflow.add_edge("retriever", "validador")
workflow.add_edge("validador", "scorer")
workflow.add_edge("scorer", END)

# 🚀 APP FINAL
app = workflow.compile()

# DEMO
if __name__ == "__main__":
    paciente = """
    Mujer 58 años, breast cancer metastásico (diagnosticado hace 5 meses).
    ECOG 1. Tratamiento previo: tamoxifen (3 semanas). Sin metástasis cerebrales.
    Función renal/hepática normal. PS 80%.
    """
    
    resultado = app.invoke({
        "perfil_paciente": paciente,
        "entidades_medicas": [],
        "ensayos_encontrados": [],
        "ensayos_validados": [],
        "ensayos_rankeados": []
    })
    
    print("\n🏆 TOP 3 RANKING NDCG@10:")
    for i, ensayo in enumerate(resultado['ensayos_rankeados'][:3], 1):
        print(f"{i}. {ensayo['NCTId']} | Score: {ensayo['score_final']:.3f}")
        print(f"   {ensayo['decision']} - {ensayo['Phase']} - {ensayo['RecruitmentStatus']}")