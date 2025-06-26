"""
Script to launch the agent manager and run the configuration.json experiment.
Execute: python launcher.py
"""

# START of import headers to use five_client without doing pip install . on the five_client package
import os
import sys
import subprocess

sys.path.insert(0, os.path.abspath("../../"))
# END

import asyncio
import traceback
import re
import json
import spade
import plotly.express as px
import pandas as pd
from datetime import datetime, timezone
import uuid
import networkx as nx
from aioxmpp import JID
from royalflush import __version__
from pathlib import Path


from typing import Any, Dict, Optional


import logging
import io
import argparse

from pathlib import Path
from aiohttp import web



from five_client.agent import AgentManager, AgentNodeBase, CoordinatorAgent, ObserverAgent, AgentFactory
from five_client.datatypes.experiment import ExperimentRawData
from five_client.datatypes.graph import GraphManager
from five_client.log.general import GeneralLogManager
from five_client.log.log import setup_loggers


manager = None
observer = None
coordinator = None
logger = None
graph_manager = None

uuid4 = None  # type: uuid.UUID | None
server = ""
algorithm = ""
algorithm_rounds = 0
consensus_iterations = 0
training_epochs = 0
graph_path = ""
dataset = ""
distribution = ""
ann = ""
max_message_size = 250_000

carga_sintetica = True

data = {}
data[1] = {
         "nAgents": int(0),
         "nActiveAgents": int(0),
         "progress": {"progress": 0, "status": "unknown", "current_round": int(0)},
         "accuracy": {"accuracy": 0, "change": 0, "loss": 0},
         "messages": {"total": 0, "status": "unknown"}
    }
training_history = []
agents_training_history = {}
connected_websockets = set()
last_round = 0
completed_rounds_cache = set()

GRAPHS_PATH = Path("royalflush_graphs")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))



async def plots_handler(request):
    figures = []
    data_canada = px.data.gapminder().query("country == 'Canada'")
    fig = px.bar(data_canada, x="year", y="pop")
    figures.append(("Title 1", fig.to_html(full_html=False)))
    figures.append(("Title 2", fig.to_html(full_html=False)))
    figures.append(("Title 3", fig.to_html(full_html=False)))
    return {"figures": figures}


async def agents_handler(request):
    global manager, graph_manager
    agentes_dict = {}
    if manager:
    
        agentes_dict[0] = {
            "jid": manager.jid,
            "pass": manager.password,
            "nAgents": len([agent for agent in manager.agents if agent.presence.is_available()]),
            "agentes":{}
        }
        for agent in manager.agents:
                jid = str(agent.jid).split("@")[0].split("_")[0]
                
                avlNeighbUsernames = [
                  agente.jid.localpart.split("@")[0].split("_")[0] for agente in manager.agents if agente.presence.is_available() and agente.jid in agent.neighbours 
                ]
                obsUsernames = [
                   observer.localpart for observer in agent.observers
                ]

                agentes_dict[0]["agentes"][jid] = {
                    "jid": str(jid),
                    "name": str(agent.jid).split("_")[0],
                    "observers": obsUsernames,
                    "neighbours": avlNeighbUsernames,
                    "available": agent.presence.is_available()
                }
    agentes_dict[0]["agentes"] = dict(
            sorted(agentes_dict[0]["agentes"].items(), key=lambda x: not x[1]["available"])
        )
    return {"agents": agentes_dict}

def update_agents_training_history():
    global agents_training_history
    agents_metrics = get_agents_accuracy()

    for agent_data in agents_metrics:
        agent = agent_data["agent"]
        if agent not in agents_training_history:
            agents_training_history[agent] = []

        round_data = {
            "round": agent_data["round"],
            "accuracy": agent_data["test_accuracy"],
            "loss": agent_data["test_loss"]
        }

        agents_training_history[agent].append(round_data)


async def websocket_handler(request):
    global data, algorithm_rounds, last_round, manager
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    connected_websockets.add(ws)

    last_state = json.dumps(data[1]) 
    try:
        while True:
            round_data = {
                "round": get_training_progress()["current_round"],
                "accuracy": get_model_accuracy()["accuracy"],
                "loss": get_model_accuracy()["loss"],
                "time": get_round_time()["round_time"]
            }
            
            if last_round != round_data["round"] and round_data["round"] > 0:
                training_history.append(round_data)
                update_agents_training_history() 
                last_round = round_data["round"]
                for agent in manager.agents:
                    agent.global_round = last_round
                               
                

            data[1] = {
                "nAgents": len(manager.agents),
                "nActiveAgents": len([agent for agent in manager.agents if agent and getattr(agent, "presence", None) and agent.presence.is_available()]),
                "progress": get_training_progress(),
                "accuracy": get_model_accuracy(),
                "messages": get_message_stats(),
                "training_history": training_history[::-1],
                "agents_training_history": agents_training_history
            }

            current_state = json.dumps(data[1])
            if current_state != last_state and not ws.closed:  
                print(f"Hay cambio y envio!")
                await ws.send_str(current_state)
                last_state = current_state
            await asyncio.sleep(3)  
    except Exception as e:
        print(f"Error: {e}")
    finally:
        connected_websockets.remove(ws)
        await ws.close()
    return ws 

async def dashboard_handler(request):
    global data, algorithm, uuid4, algorithm_rounds, consensus_iterations, training_epochs, server, graph_path, dataset, distribution, ann
    data[0] = {
        "algorithm": algorithm,
        "uuid4": uuid4,
        "algorithm_rounds": algorithm_rounds,
        "consensus_iterations": consensus_iterations,
        "training_epochs": training_epochs,
        "server": server,
        "graph_path": graph_path,
        "dataset": dataset,
        "distribution": distribution,
        "ann": ann
    }

   
    
    return {"data": data}


async def stop_agents_handler(request):
    await manager.stop_agents()
    return {}

async def stop_handler(request):
    await manager.stop()
    return {}


async def pause_singleAgent_handler(request):
    try:
        data = await request.json()
        jid = data.get("jid")
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    await manager.pause_agent(jid)
    return web.json_response({"success": True, "jid": jid}) 

async def resume_singleAgent_handler(request):
    try:
        data = await request.json()
        jid = data.get("jid")
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    await manager.resume_agent(jid)
    print(manager.agents)
    return web.json_response({"success": True, "jid": jid})

async def load_graph_controller(request):
        # WARNING: don't do that if you plan to receive large files!
        data = await request.post()

        if 'stopMAS' in data:
            manager.stop()
            raise web.HTTPFound('/manager/launcher')
        else:
            global algorithm, uuid4, algorithm_rounds, consensus_iterations, training_epochs, server, graph_path, dataset, distribution, ann
            algorithm = data.get('algorithm', '')
            uuid4 = "generate_new_uuid4" if data.get('generate_new_uuid4','') else ""
            algorithm_rounds = data.get('algorithm_rounds',0)
            consensus_iterations = data.get('consensus_iterations',0)
            training_epochs = data.get('training_epochs',0)
            graph_path = data.get('graph_path','')
            dataset = data.get('dataset','')
            distribution = data.get('distribution','')
            ann = data.get('ann','')
            graph_file_data = data['graphInputFile']
            await load_graph(graph_file_data)


            raise web.HTTPFound('/manager/dashboard')
        
async def gen_graph_controller(request):
        # WARNING: don't do that if you plan to receive large files!
        gm = GraphManager()
        data = await request.post()
        nodesNumber = int(data.get('nodesNumber', '0'))
        fileName = data.get('graphOutputFile', '')
        graphType = data.get('graphGenerator', '')

        out = "../royalflush_graphs/" + f"{nodesNumber:03}_agents_"

        agents = [JID.fromstr(f"a{i}@localhost") for i in range(nodesNumber)]
        if graphType == 'complete':
            gm.generate_complete(agents)
            out = out + "complete"
        elif graphType == 'ring':
            gm.generate_ring(agents)
            out = out + "ring"
        elif graphType == 'small-world':
            gm.generate_small_world(agents, k=4, p=0.3)
            out = out + "sw"

        if fileName != '':
            out = "../royalflush_graphs/" + fileName

        os.makedirs(os.path.dirname(out), exist_ok=True)

        try:
            gm.export_to_gml(f"{out}.gml")
        except Exception as e:
            return web.json_response({"message": "Error generating graph", "error": str(e)})
        raise web.HTTPFound('/manager/launcher')

async def get_graph_data(request):
        try:
            print("API call received")  # Debug
            graph_data = graph_manager.export_as_cytoscape_json()
            print(f"Graph data: {graph_data}")  # Debug
            return web.json_response(graph_data)
        except Exception as e:
            print(f"Error in get_graph_data: {e}")  # Debug
            import traceback
            traceback.print_exc()
            return web.json_response({"error": str(e)}, status=500)
    
async def update_graph_handler(request):
    global manager, graph_manager
    data = await request.json()

    action = data.get('action')
    payload = data.get('data')

    source = payload.get('source')
    target = payload.get('target')

    agents = await manager.update_neighbours(source, target, action)
    all_synced = True
    
    print("agentes: " + source + target)
    for agent in agents:
        if await agent.sync_neighbours_subscription():
            print(f"Topology updated successfully for {agent}")
        else:
            print(f"Failed to sync subscriptions for {agent}")
            all_synced = False

    if all_synced:
        if action == "create_edge":
            graph_manager.graph.add_edge(source, target)
        elif action == "delete_edge":
            graph_manager.graph.remove_edge(source, target)
        print("Graph topology updated successfully")
    else:
        print("Graph not updated due to sync failures")

    


    agentes_dict = {}
    if manager:
    
        agentes_dict[0] = {
            "jid": manager.jid,
            "pass": manager.password,
            "nAgents": len([agent for agent in manager.agents if agent.presence.is_available()]),
            "agentes":{}
        }
        for agent in manager.agents:
                jid = str(agent.jid).split("@")[0].split("_")[0]
                
                avlNeighbUsernames = [
                  agente.jid.localpart.split("@")[0].split("_")[0] for agente in manager.agents if agente.presence.is_available() and agente.jid in agent.neighbours 
                ]
                obsUsernames = [
                   observer.localpart for observer in agent.observers
                ]

                agentes_dict[0]["agentes"][jid] = {
                    "jid": str(jid),
                    "name": str(agent.jid).split("_")[0],
                    "observers": obsUsernames,
                    "neighbours": avlNeighbUsernames,
                    "available": agent.presence.is_available()
                }
    agentes_dict[0]["agentes"] = dict(
            sorted(agentes_dict[0]["agentes"].items(), key=lambda x: not x[1]["available"])
        )
    return web.json_response(data)

        
##################################################################################################################

def setup_web(manager: AgentManager, web_address: str, web_port: int) -> None:
    manager.web.add_get("/manager/launcher", dashboard_handler, "html/launcher.html")
    manager.web.add_get("/manager/dashboard", dashboard_handler, "html/dashboard.html")
    manager.web.add_get("/manager/project", plots_handler, "html/project.html")
    manager.web.add_get("/manager/agents", agents_handler, "html/agents.html")
    manager.web.add_get("/manager/plots", plots_handler, "html/plots.html")
    manager.web.add_get("/manager/stop_all_agents", stop_agents_handler, "html/dashboard.html")
    manager.web.add_get("/manager/stop", stop_handler, "html/agents.html")
    manager.web.add_post("/manager/pause_agent", pause_singleAgent_handler, "html/agents.html")
    manager.web.add_post("/manager/resume_agent", resume_singleAgent_handler, "html/agents.html")
    manager.web.add_post("/api/graph-update", update_graph_handler, "html/agents.html")
    manager.web.add_post("/submit", load_graph_controller, "html/dashboard.html")
    manager.web.add_post("/submit_graph", gen_graph_controller, "html/dashboard.html")
    manager.web.app.router.add_get("/ws", websocket_handler)
    manager.web.add_get('/api/graph-data', get_graph_data, "html/agents.html")
    manager.web.app.router.add_static('/static', os.path.join(os.path.dirname(__file__), '../web/rfLogo'))
    # manager.web.add_get("/manager/plugins", plots, "html/plugins.html")
    manager.web.start(
        hostname=web_address,
        port=web_port,
        templates_path="../web/templates",
    )
    print(f"Manager started: http://{web_address}:{web_port}/manager/dashboard")

##################################################################################################################
## Manejo Logs ##
def get_latest_log_folder():
    """Encuentra la carpeta de logs más reciente."""
    log_base = Path("logs")
    if not log_base.exists():
        return None
    
    # Carpetas con formato de fecha o UUID
    date_pattern = r"\d{4}_\d{2}_\d{2}_T_\d{2}_\d{2}_\d{2}_\d+_Z"
    folders = []
    
    for folder in log_base.iterdir():
        if folder.is_dir():
            if re.match(date_pattern, folder.name):
                raw_folder = folder / "raw"
                if raw_folder.exists() and raw_folder.is_dir():
                    folders.append(raw_folder)
            elif all(c.isalnum() or c == '_' for c in folder.name):
                raw_folder = folder / "raw"
                if raw_folder.exists() and raw_folder.is_dir():
                    folders.append(raw_folder)
    
    if not folders:
        return None
    folders.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return folders[0]

def get_training_progress():
    """Progreso general del entrenamiento."""
    global algorithm_rounds, manager, completed_rounds_cache
    log_folder = get_latest_log_folder()
    if not log_folder:
        return {"progress": 0, "status": "unknown", "current_round": int(0)}
    
    nn_inference_log = log_folder / "nn_inference.csv"
    nn_convergence_log = log_folder / "nn_convergence.csv"
    
    total_rounds = int(algorithm_rounds)
    current_round = 0
    status = "unknown"
    
    if nn_inference_log.exists():
        try:
            df = pd.read_csv(nn_inference_log)
            expected_columns = ['log_timestamp', 'log_name', 'algorithm_round']
            if len(df.columns) >= len(expected_columns):
                df.columns = expected_columns + list(df.columns[len(expected_columns):])
                if len(df) > 0:
                    num_agents = len([agente for agente in manager.agents if agente.presence.is_available()])

                    round_counts = df.groupby('algorithm_round')['agent'].nunique()
                    completed_rounds = round_counts[round_counts == num_agents & (~round_counts.index.isin(completed_rounds_cache))]
                    if not completed_rounds.empty:
                        current_round = completed_rounds.index.max()
                        completed_rounds_cache.update(completed_rounds.index)
                    
                    # Estado basado en tiempo entre rondas
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        if len(df) > 1:
                            # Tiempo medio entre rondas
                            grouped = df.groupby('algorithm_round')['timestamp'].min().reset_index()
                            grouped = grouped.sort_values('algorithm_round')
                            
                            if len(grouped) > 1:
                                # Diferencias de tiempo entre rondas consecutivas
                                grouped['next_timestamp'] = grouped['timestamp'].shift(-1)
                                grouped['time_diff'] = (grouped['next_timestamp'] - grouped['timestamp']).dt.total_seconds()
                                
                                # Tiempo medio entre rondas (excluyendo valores nulos)
                                avg_time_between_rounds = grouped['time_diff'].dropna().mean()
                                
                                # Tiempo desde la última actualización
                                last_update = grouped['timestamp'].max()
                                time_since_last = (datetime.now(tz=timezone.utc) - pd.to_datetime(last_update)).total_seconds()
                                
                                # Si tiempo desde la última actualización es mayor que 2 veces el promedio --> retrasado
                                if time_since_last > (2 * avg_time_between_rounds) and avg_time_between_rounds > 0:
                                    status = "delayed"
                                else:
                                    status = "on schedule"
        except Exception as e:
            print(f"Error procesando nn_inference.csv para progress: {e}")
    
    if current_round == 0 and nn_convergence_log.exists():
        try:
            df = pd.read_csv(nn_convergence_log)
            expected_columns = ['log_timestamp', 'log_name', 'algorithm_round']
            if len(df.columns) >= len(expected_columns):
                df.columns = expected_columns + list(df.columns[len(expected_columns):])
                if len(df) > 0:
                    current_round = df['algorithm_round'].max()
        except Exception as e:
            print(f"Error procesando nn_convergence.csv para progress: {e}")
    
    # Calcular progreso
    progress = (current_round / total_rounds) * 100 if total_rounds > 0 else 0
    
    return {"progress": float(round(progress, 1)), "status": status, "current_round": int(current_round)}

def get_model_accuracy():
    """Precisión actual del modelo y su evolución."""
    log_folder = get_latest_log_folder()
    if not log_folder:
        return {"accuracy": 0, "change": 0, "loss": 0}
    
    nn_inference_log = log_folder / "nn_inference.csv"
    if not nn_inference_log.exists():
        return {"accuracy": 0, "change": 0, "loss": 0}
    
    try:
        df = pd.read_csv(nn_inference_log)
        
        expected_columns = [
            'log_timestamp', 'log_name', 'algorithm_round', 'timestamp', 'agent', 
            'seconds_to_complete', 'epochs', 'mean_training_accuracy', 
            'mean_training_loss', 'validation_accuracy', 'validation_loss', 
            'test_accuracy', 'test_loss'
        ]
        
        if len(df.columns) >= len(expected_columns):
            df.columns = expected_columns[:len(df.columns)]
            
            df = df.sort_values('algorithm_round')
            
            if len(df) > 0 and 'test_accuracy' in df.columns and 'test_loss' in df.columns:
                # Agrupar por ronda para obtener la precisión media por ronda
                metrics_by_round = df.groupby('algorithm_round')[['test_accuracy', 'test_loss']].mean().reset_index()
                
                # Última precisión y pérdida
                latest_accuracy = metrics_by_round['test_accuracy'].iloc[-1] * 100  # Convertir a porcentaje
                latest_loss = metrics_by_round['test_loss'].iloc[-1]
                
                # Cambio desde la ronda anterior
                if len(metrics_by_round) > 1:
                    previous_accuracy = metrics_by_round['test_accuracy'].iloc[-2] * 100
                    change = latest_accuracy - previous_accuracy
                else:
                    change = latest_accuracy - 0
                    
                return {
                    "accuracy": float(round(latest_accuracy, 1)), 
                    "change": float(round(change, 1)),
                    "loss": float(round(latest_loss, 4))
                }
    except Exception as e:
        print(f"Error procesando nn_inference.csv para accuracy: {e}")
    
    return {"accuracy": 0, "change": 0, "loss": 0}

def get_agents_accuracy():
    """Accuracy y loss de todas las rondas nuevas por agente."""
    log_folder = get_latest_log_folder()
    if not log_folder:
        return []

    nn_inference_log = log_folder / "nn_inference.csv"
    if not nn_inference_log.exists():
        return []

    try:
        df = pd.read_csv(nn_inference_log)

        expected_columns = [
            'log_timestamp', 'log_name', 'algorithm_round', 'timestamp', 'agent',
            'seconds_to_complete', 'epochs', 'mean_training_accuracy',
            'mean_training_loss', 'validation_accuracy', 'validation_loss',
            'test_accuracy', 'test_loss'
        ]

        if len(df.columns) >= len(expected_columns):
            df.columns = expected_columns[:len(df.columns)]

            agents_metrics = []
            for _, row in df.iterrows():
                agent = row["agent"]
                round_number = int(row["algorithm_round"])

                if agent not in agents_training_history:
                    agents_training_history[agent] = []

                # Añadir si ronda aún no está guardada
                existing_rounds = {entry["round"] for entry in agents_training_history[agent]}
                if round_number not in existing_rounds:
                    agents_metrics.append({
                        "agent": agent,
                        "round": round_number,
                        "test_accuracy": float(round(row["test_accuracy"] * 100, 1)),
                        "test_loss": float(round(row["test_loss"], 4))
                    })

            return agents_metrics
    except Exception as e:
        print(f"Error procesando nn_inference.csv para métricas por agente: {e}")

    return []




def get_message_stats():
    """Estadísticas sobre mensajes intercambiados."""
    log_folder = get_latest_log_folder()
    if not log_folder:
        return {"total": 0, "status": "unknown"}
    
    message_log = log_folder / "message.csv"
    if not message_log.exists():
        return {"total": 0, "status": "unknown"}
    
    try:
        df = pd.read_csv(message_log)
        total_messages = len(df)
        
        status = "healthy"
        
        if 'seconds_to_complete' in df.columns:
            avg_time = df['seconds_to_complete'].mean()
            if avg_time > 5.0:  # Umbral
                status = "issues detected"
        elif 'status' in df.columns:
            error_count = df[df['status'].str.contains('error|timeout|failed', case=False, na=False)].shape[0]
            error_rate = error_count / total_messages if total_messages > 0 else 0
            if error_rate > 0.05:  # Más del 5% de errores
                status = "issues detected"
        
        return {"total": int(total_messages), "status": status}
    except Exception as e:
        print(f"Error procesando message.csv: {e}")
    
    return {"total": 0, "status": "unknown"}

def get_round_time():
    """Tiempo de ejecución de cada ronda del algoritmo desde algorithm.csv."""
    log_folder = get_latest_log_folder()
    if not log_folder:
        return {"round": 0, "round_time": 0, "total_time": 0, "average_time": 0}

    algorithm_log = log_folder / "algorithm.csv"
    if not algorithm_log.exists():
        return {"round": 0, "round_time": 0, "total_time": 0, "average_time": 0}

    try:
        df = pd.read_csv(algorithm_log)

        if df.empty:
            return {"round": 0, "round_time": 0, "total_time": 0, "average_time": 0}

        expected_columns = ["algorithm_round", "seconds_to_complete", "agent"]
        if not all(col in df.columns for col in expected_columns):
            print(f"Faltan columnas en algorithm.csv. Columnas esperadas: {expected_columns}")
            return {"round": 0, "round_time": 0, "total_time": 0, "average_time": 0}

        # Ordenar por ronda
        df = df.sort_values("algorithm_round")

        # Tiempo promedio por ronda (en caso de que haya múltiples agentes por ronda)
        round_times = df.groupby("algorithm_round")["seconds_to_complete"].mean().reset_index()

        if round_times.empty:
            return {"round": 0, "round_time": 0, "total_time": 0, "average_time": 0}

        # Última ronda y su tiempo de forma segura
        latest_round = round_times["algorithm_round"].max()

        filtered_rounds = round_times[round_times["algorithm_round"] == latest_round]
        if len(filtered_rounds) > 0:
            latest_round_time = filtered_rounds["seconds_to_complete"].iloc[0]
        else:
            latest_round_time = 0

        # Tiempo total acumulado
        total_time = round_times["seconds_to_complete"].sum()

        # Tiempo promedio de todas las rondas
        average_time = round_times["seconds_to_complete"].mean()

        return {
            "round": int(latest_round),
            "round_time": float(round(latest_round_time, 2)),  # Tiempo en segundos de la última ronda
            "total_time": float(round(total_time, 2)),         # Tiempo total acumulado en segundos
            "average_time": float(round(average_time, 2))      # Tiempo promedio por ronda
        }

    except Exception as e:
        print(f"Error procesando algorithm.csv para tiempo de ronda: {e}")
        return {"round": 0, "round_time": 0, "total_time": 0, "average_time": 0}
##################################################################################################################

async def load_graph(graph_data):         
        #     i = i + 1
        global algorithm, uuid4, algorithm_rounds, consensus_iterations, training_epochs, server, graph_path, dataset, distribution, ann, graph_manager, carga_sintetica
        graph_path = os.path.join(BASE_DIR, graph_path)
        data = {
            "uuid4": uuid4,
            "algorithm": algorithm,
            "algorithm_rounds": int(algorithm_rounds),
            "consensus_iterations": int(consensus_iterations),
            "training_epochs": int(training_epochs),
            "xmpp_domain": server,
            "graph_path": graph_path,
            "dataset": dataset,
            "distribution": distribution,
            "ann": ann
        }
        experiment = ExperimentRawData(data)
        logger.info(f"Experiment details: {repr(experiment)}")

        # UUID4
        if experiment.uuid4 == "generate_new_uuid4":
            uuid4 = uuid.uuid4()
        elif isinstance(experiment.uuid4, str):
            uuid4 = uuid.UUID(experiment.uuid4)

        # Graph
        logger.debug("Initializating GraphManager...")
        graph_manager = GraphManager()
        graph_manager.import_from_gml(experiment.graph_path)
        logger.debug(f"Graph {experiment.graph_path} loaded")

        
       
        agent_factory = AgentFactory(
            experiment=experiment,
            graph_manager=graph_manager,
            coordinator_jid=JID.fromstr(str(coordinator.jid)),
            observer_jids=[JID.fromstr(str(observer.jid))],
            uuid=uuid4,
            max_message_size=max_message_size
        )

        coordinator.coordinated_agents=graph_manager.list_agents_jids(uuid=uuid4)
        await coordinator.behaviour()

        manager.agents = agent_factory.create_agents(carga_sintetica)

        

async def generate_graph_file(data):
        # WARNING: don't do that if you plan to receive large files!
        nodesNumber = int(data['nodesNumber'])

        if data['graphGenerator'] == 'complete':
            nxOutputGraph = nx.complete_graph(nodesNumber)
        elif data['graphGenerator'] == "cycle":
            nxOutputGraph = nx.cycle_graph(nodesNumber)
        elif data['graphGenerator'] == "wheel":
            nxOutputGraph = nx.wheel_graph(nodesNumber)
        elif data['graphGenerator'] == "star":
            nxOutputGraph = nx.star_graph(nodesNumber)
        elif data['graphGenerator'] == "random":
            prob = float(data['probability'])
            nxOutputGraph = nx.erdos_renyi_graph(nodesNumber, prob)
        elif data['graphGenerator'] == "watts":
            prob = float(data['probability'])
            k = int(data['neighDist']) # neighbours distance
            nxOutputGraph = nx.watts_strogatz_graph(nodesNumber, k, prob)
        elif data['graphGenerator'] == "barabasi":
            m = int(data['edgesNew']) # number of edges to attach a new node
            nxOutputGraph = nx.barabasi_albert_graph(nodesNumber, m)
        elif data['graphGenerator'] == "geometric":
            radius = float(data['radius']) # distance threshold value
            nxOutputGraph = nx.random_geometric_graph(nodesNumber, radius)

        # writing gml file
        graph_buffer = io.StringIO()
        
        graph_buffer.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        graph_buffer.write('<graphml xmlns="http://graphml.graphdrawing.org/xmlns" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" ')
        graph_buffer.write('xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.1/graphml.xsd">\n')
        graph_buffer.write('<graph id="G" edgedefault="undirected">\n')

        # Escribir nodos
        for i, node in enumerate(nxOutputGraph.nodes()):
            graph_buffer.write('<node id="node_{}"/>\n'.format(i+1))

        # Escribir relaciones (edges)
        for u, v in nxOutputGraph.edges():
            graph_buffer.write('<edge source="node_{}" target="node_{}"/>\n'.format(u+1, v+1))

        graph_buffer.write('</graph>\n')
        graph_buffer.write('</graphml>\n')

        graph_buffer.seek(0)  # Volver al inicio del buffer
        return graph_buffer


def register_user(username, domain, password):
    container = "ejabberd"

    # Comprobamos si el usuario ya existe
    result = subprocess.run(
        ["docker", "exec", container, "ejabberdctl", "registered_users", domain],
        capture_output=True, text=True
    )

    if username in result.stdout:
        print(f"[INFO] Usuario {username}@{domain} ya registrado.")
        return

    # Si no está, lo registramos
    subprocess.run(
        ["docker", "exec", container, "ejabberdctl", "register", username, domain, password],
        check=True
    )
    print(f"[INFO] Usuario {username}@{domain} registrado correctamente.")



async def main():
    # try:# DEBUG
    print("FIVE version: ", five_client.__version__)
    print("SPADE version: ", spade.__version__)

    global manager
    global coordinator
    global observer
    global server
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--server", help="XMPP server", required=True)
    args = parser.parse_args()
    server = args.server

    print(f"XMPP server: {server}")
    
    global logger
    logger = GeneralLogManager(extra_logger_name="main")
    logger.info("Starting...")
    logger.info(f"Royal FLush version: {__version__}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"SPADE version: {spade.__version__}")

    logger.debug(f"Initializating coordinator...")
    coordinator = CoordinatorAgent(
        jid = str(f"coordinator@{server}"),
        password = "coord",
        max_message_size=max_message_size, 
        coordinated_agents=None,
        verify_security=False,
    )
    register_user(str("coordinator"), str(server),  "coord")
    await asyncio.sleep(0.2)    

    observer = ObserverAgent(
        jid = str(f"observer1@{server}"),
        password = "observ",
        max_message_size=max_message_size,
        verify_security=False,
    )
    register_user(str("observer1"), str(server),  "observ")

    manager = AgentManager(
        jid=str(f"manager_marcram@{server}"),
        password="mrg",
        agents=[],
        max_message_size=max_message_size, 
        agents_coordinator=JID.fromstr(str(coordinator.jid)),
        agents_observers = [JID.fromstr(str(observer.jid))],
    )
    register_user(str("manager_marcram"), str(server),  "mrg")
    setup_web(manager=manager, web_address="127.0.0.1", web_port=10000)

    try:
        logger.info("Starting observers...")

        await observer.start(auto_register=True)
        await asyncio.sleep(0.2)
        logger.info("Observers initialized.")

        logger.info("Starting coordinator...")
        await coordinator.start(auto_register=True)
        await asyncio.sleep(0.2)
        logger.info("Coordinator initialized.")

        logger.info("Starting launcher...")
        await manager.start(auto_register=True)
        await asyncio.sleep(0.2)
        logger.info("Launcher initialized.")

        await asyncio.sleep(5)
        while not coordinator.ready_to_start_algorithm or any(ag.is_alive() for ag in manager.agents):
            await asyncio.sleep(5)
    
    except KeyboardInterrupt as e:
        raise e

    except Exception as e:
        logger.exception(e)
        traceback.print_exc()

    finally:
        logger.info("Stopping...")
        if coordinator.is_alive():
            await coordinator.stop()
        if manager.is_alive():
            await manager.stop()
    
        logger.info("Run finished.")

if __name__ == "__main__":
    import asyncio
    try:
        setup_loggers(general_level=logging.INFO)
        spade.run(main())
    except KeyboardInterrupt:
        pass
    except Exception:
        traceback.print_exc()
