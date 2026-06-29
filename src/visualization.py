# visualization.py – полностью исправленная версия
from pyvis.network import Network
import json
from collections import defaultdict

def visualize_graph(G, results, users, node_types=None, blue_agent=None, output_path="network_visualization_pro.html"):
    print("\n🔍 Generating visualization...")

    net = Network(height="800px", width="100%", directed=True, bgcolor="#ffffff", font_color="black")
    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -9000,
          "springLength": 260,
          "springConstant": 0.03,
          "damping": 0.09
        }
      },
      "edges": { "smooth": { "type": "continuous" } },
      "interaction": { "hover": true, "tooltipDelay": 100 }
    }
    """)

    timeline = results.get("timeline", [])
    users_final = results.get("users_final", {})
    states_history = results.get("states_history", [])

    # Сбор сообщений для узлов
    node_received_messages = defaultdict(list)
    node_sent_messages = defaultdict(list)

    for event in timeline:
        to_node = event.get("to")
        if to_node is not None and to_node != -1:
            node_received_messages[to_node].append({
                "t": event.get("t", 0),
                "from": event.get("from"),
                "text": event.get("text", ""),
                "category": event.get("category", "unknown"),
                "h": event.get("h", 0),
                "risk_score": event.get("detected_risk", event.get("blue_risk_score", 0)),
                "risk_level": event.get("detected_category", event.get("blue_risk_level", "UNKNOWN"))
            })

        from_node = event.get("from")
        if from_node is not None:
            if (event.get("to") is None and
                not event.get("category", "").startswith("detected") and
                event.get("category") != "warning"):
                node_sent_messages[from_node].append({
                    "t": event.get("t", 0),
                    "to": event.get("to"),
                    "text": event.get("text", ""),
                    "category": event.get("category", "unknown"),
                    "h": event.get("h", 0)
                })

    # Начальные состояния
    users_initial = {}
    for k, v in users.items():
        if hasattr(v, 'b'):
            users_initial[str(k)] = {'b': v.b, 'c': v.c, 'e': v.e}
        elif isinstance(v, dict):
            users_initial[str(k)] = v
        else:
            users_initial[str(k)] = {'b': 0, 'c': 0, 'e': 0}

    # История состояний для обычных узлов
    node_full_history = {}
    for node in G.nodes():
        node_str = str(node)
        history = []
        if node_str in users_initial:
            history.append({
                "t": 0,
                "b": users_initial[node_str].get('b', 0),
                "c": users_initial[node_str].get('c', 0),
                "e": users_initial[node_str].get('e', 0)
            })
        for step, snapshot in enumerate(states_history, start=1):
            if node_str in snapshot:
                history.append({
                    "t": step,
                    "b": snapshot[node_str]["b"],
                    "c": snapshot[node_str]["c"],
                    "e": snapshot[node_str]["e"]
                })
        if not history and node_str in users_final:
            history.append({
                "t": len(states_history),
                "b": users_final[node_str].get('b', 0),
                "c": users_final[node_str].get('c', 0),
                "e": users_final[node_str].get('e', 0)
            })
        node_full_history[node_str] = history

    # Рёберные передачи
    message_transmissions = {}
    for ev in timeline:
        key = (ev["from"], ev["to"])
        message_transmissions.setdefault(key, []).append(ev)

    # Добавление узлов в граф
    for node in G.nodes():
        node_str = str(node)
        tooltip_lines = [f"━━━━━━━━━━━━━━━━━━━━━━\n🔷 AGENT {node}\n━━━━━━━━━━━━━━━━━━━━━━"]

        ntype = None
        if node_types:
            ntype = node_types.get(node) or node_types.get(node_str)

        if ntype == "U":
            color, shape, base_size = "#808080", "circle", 40
            tooltip_lines.append("📌 TYPE: USER (Серый круг)")
        elif ntype == "R":
            color, shape, base_size = "#e74c3c", "box", 40
            tooltip_lines.append("📌 TYPE: RED AGENT (Красный квадрат)")
        elif ntype == "L":
            color, shape, base_size = "#f1c40f", "triangle", 25
            tooltip_lines.append("📌 TYPE: LLM AGENT (Жёлтый треугольник)")
        elif ntype == "B":
            color, shape, base_size = "#3498db", "diamond", 25
            tooltip_lines.append("📌 TYPE: BLUE MODERATOR (Синий ромб)")
        else:
            color, shape, base_size = "#808080", "circle", 35
            tooltip_lines.append("📌 TYPE: USER (Автоопределен)")

        if node_str in users_initial:
            init = users_initial[node_str]
            tooltip_lines.append("📊 INITIAL STATE:")
            tooltip_lines.append(f"   • b: {init.get('b',0):.4f}")
            tooltip_lines.append(f"   • c: {init.get('c',0):.4f}")
            tooltip_lines.append(f"   • e: {init.get('e',0):.4f}")

        if node_str in users_final and ntype not in ["B", "L", "R"]:
            tooltip_lines.append("━━━━━━━━━━━━━━━━━━━━━━")
            fin = users_final[node_str]
            tooltip_lines.append("📊 FINAL STATE:")
            tooltip_lines.append(f"   • b: {fin.get('b',0):.4f}")
            tooltip_lines.append(f"   • c: {fin.get('c',0):.4f}")
            tooltip_lines.append(f"   • e: {fin.get('e',0):.4f}")

        tooltip_lines.append("━━━━━━━━━━━━━━━━━━━━━━")
        sent = sum(1 for e in timeline if e["from"] == node and e.get("to") is None and
                   not e.get("category", "").startswith("detected") and e.get("category") != "warning")
        recv = sum(1 for e in timeline if e["to"] == node)
        tooltip_lines.append(f"📤 SENT (original): {sent}")
        tooltip_lines.append(f"📥 RECEIVED: {recv}")
        tooltip_lines.append("━━━━━━━━━━━━━━━━━━━━━━")
        tooltip_lines.append(f"🔗 OUT: {G.out_degree(node)}")
        tooltip_lines.append(f"🔗 IN: {G.in_degree(node)}")
        tooltip_lines.append("━━━━━━━━━━━━━━━━━━━━━━")
        tooltip_lines.append("💡 Двойной клик → подробная информация")

        net.add_node(
            node,
            label=str(node),
            color=color,
            shape=shape,
            size=base_size,
            scaling={"min": 10, "max": 100},
            title="\n".join(tooltip_lines),
            font={"size": 14, "color": "black", "face": "Arial"}
        )

    # Добавление рёбер
    edge_id = 0
    edge_map = {}
    edge_full_info = {}
    edge_messages_by_time = {}

    for u, v, data in G.edges(data=True):
        msgs = message_transmissions.get((u, v), [])
        full_msgs = []
        for m in msgs:
            full_msgs.append(
                f"<div style='border-bottom:1px solid #ddd; padding:6px; margin-bottom:4px; font-size:11px;'>"
                f"<b>━━━ Message t={m.get('t',0)} ━━━</b><br>"
                f"<b>📝 Text:</b> {m.get('text', '')}<br>"
                f"<b>🎯 Category:</b> {m.get('category', 'unknown')}<br>"
                f"<b>📊 h:</b> {m.get('h', 0):.4f}<br>"
                f"<b>⏰ Age:</b> {m.get('age', 0)} steps<br>"
                f"<b>🤖 Blue Risk:</b> {m.get('blue_risk_score', 0):.4f}<br>"
                f"<b>⚠️ Level:</b> {m.get('blue_risk_level', 'UNKNOWN')}<br></div>"
            )
        edge_full_info[edge_id] = {"u": u, "v": v, "messages": full_msgs, "total": len(full_msgs)}

        by_time = {}
        for m in msgs:
            t = m.get('t', 0)
            by_time.setdefault(t, []).append({
                "text": m.get('text', '')[:80],
                "category": m.get('category', 'unknown'),
                "h": m.get('h', 0),
                "age": m.get('age', 0),
                "blue_risk_score": m.get('blue_risk_score', 0),
                "blue_risk_level": m.get('blue_risk_level', 'UNKNOWN')
            })
        edge_messages_by_time[edge_id] = by_time

        if msgs:
            max_h = max(m.get('h', 0) for m in msgs)
            if max_h > 0.5:
                edge_color = "#e74c3c"
            elif max_h > 0.2:
                edge_color = "#ff9800"
            elif max_h > 0:
                edge_color = "#ffeb3b"
            else:
                edge_color = "#3498db"
        else:
            edge_color = "#95a5a6"

        reposts = data.get('reposts', 0)
        risk_sum = data.get('risk_sum', 0)
        avg_risk = risk_sum / reposts if reposts > 0 else 0

        net.add_edge(u, v, id=edge_id, color=edge_color, width=2, arrows="to",
                     title=f"EDGE {u}→{v}\n📊 Reposts: {reposts}\n📈 Risk Sum: {risk_sum}\n🎯 Avg Risk: {avg_risk:.3f}\n💡 Двойной клик → полная информация")
        edge_map[f"{u},{v}"] = edge_id
        edge_id += 1

    # Таймлайн
    timeline_by_time = {}
    for e in timeline:
        t = e["t"]
        timeline_by_time.setdefault(t, []).append(e)
    max_time = max(timeline_by_time.keys()) if timeline_by_time else 0

    # Сериализация данных для JS
    timeline_json = json.dumps(timeline_by_time, ensure_ascii=False)
    edge_map_json = json.dumps(edge_map)
    edge_full_info_json = json.dumps(edge_full_info, ensure_ascii=False)
    edge_messages_by_time_json = json.dumps(edge_messages_by_time, ensure_ascii=False)
    node_history_json = json.dumps(node_full_history, ensure_ascii=False)
    node_received_json = json.dumps({str(k): v for k, v in node_received_messages.items()}, ensure_ascii=False)
    node_sent_json = json.dumps({str(k): v for k, v in node_sent_messages.items()}, ensure_ascii=False)
    node_types_json = json.dumps({str(k): v for k, v in node_types.items()}, ensure_ascii=False)

    html = net.generate_html()

    custom_js = f"""
<script>
let timelineData = {timeline_json};
let edgeMap = {edge_map_json};
let edgeFullInfo = {edge_full_info_json};
let edgeMessagesByTime = {edge_messages_by_time_json};
let nodeStatesHistory = {node_history_json};
let nodeReceivedMessages = {node_received_json};
let nodeSentMessages = {node_sent_json};
let nodeTypes = {node_types_json};

let currentTime = 0;
let animInterval = null;
let settingsVisible = false;

function showFullInfoNode(nodeId) {{
    let modal = document.getElementById("nodeModal");
    let content = document.getElementById("nodeModalContent");
    if (!modal) {{
        let modalHtml = `
        <div id="nodeModal" style="display:none; position:fixed; z-index:10000; left:0; top:0;
             width:100%; height:100%; background:rgba(0,0,0,0.6); backdrop-filter:blur(5px);">
            <div style="position:absolute; left:50%; top:50%; transform:translate(-50%,-50%);
                 background:white; border-radius:10px; width:60%; max-width:800px; max-height:75%;
                 box-shadow:0 5px 20px rgba(0,0,0,0.3); display:flex; flex-direction:column;">
                <div style="padding:10px 15px; border-bottom:1px solid #eee; display:flex; 
                     justify-content:space-between; align-items:center;">
                    <h3 style="margin:0; font-size:16px;">🔷 NODE ${{nodeId}}</h3>
                    <button onclick="closeModal('nodeModal')" style="background:none; border:none; font-size:22px; cursor:pointer;">&times;</button>
                </div>
                <div id="nodeModalContent" style="padding:12px; overflow-y:auto; flex:1; font-family:monospace; font-size:11px;"></div>
            </div>
        </div>`;
        document.body.insertAdjacentHTML('beforeend', modalHtml);
        modal = document.getElementById("nodeModal");
        content = document.getElementById("nodeModalContent");
    }}

    let nodeType = nodeTypes[String(nodeId)] || "U";
    let html = "";

    if (nodeType === "B") {{
        let received = nodeReceivedMessages[String(nodeId)] || [];
        html = `<h4 style="margin:0 0 8px 0;">🔷 BLUE MODERATOR - Полученные сообщения</h4>`;
        if (received.length === 0) {{
            html += "<p>Нет полученных сообщений</p>";
        }} else {{
            html += `<table style="width:100%; border-collapse:collapse; font-size:10px;">
                        <thead>
                            <tr style="background:#f0f0f0; border-bottom:1px solid #ccc;">
                                <th style="padding:3px;">t</th><th style="padding:3px;">От</th><th style="padding:3px;">Текст</th><th style="padding:3px;">Кат.</th><th style="padding:3px;">h</th><th style="padding:3px;">Blue Risk</th><th style="padding:3px;">Уровень</th>
                            </tr>
                        </thead>
                        <tbody>`;
            for (let msg of received) {{
                html += `<tr style="border-bottom:1px solid #ddd;">
                            <td style="padding:3px;">${{msg.t}}</td>
                            <td style="padding:3px;">${{msg.from}}</td>
                            <td style="padding:3px;">${{msg.text.substring(0, 50)}}</td>
                            <td style="padding:3px;">${{msg.category}}</td>
                            <td style="padding:3px;">${{msg.h.toFixed(2)}}</td>
                            <td style="padding:3px;">${{msg.risk_score.toFixed(2)}}</td>
                            <td style="padding:3px;">${{msg.risk_level}}</td>
                          </tr>`;
            }}
            html += `</tbody>`;
        }}
    }} else if (nodeType === "L" || nodeType === "R") {{
        let sent = nodeSentMessages[String(nodeId)] || [];
        let title = (nodeType === "L") ? "🟡 LLM AGENT - Сгенерированные сообщения" : "🔴 RED AGENT - Сгенерированные сообщения";
        html = `<h4 style="margin:0 0 8px 0;">${{title}}</h4>`;
        if (sent.length === 0) {{
            html += "<p>Нет сгенерированных сообщений</p>";
        }} else {{
            html += `<table style="width:100%; border-collapse:collapse; font-size:10px;">
                        <thead>
                            <tr style="background:#f0f0f0; border-bottom:1px solid #ccc;">
                                <th style="padding:3px;">t</th><th style="padding:3px;">Кому</th><th style="padding:3px;">Текст</th><th style="padding:3px;">Кат.</th><th style="padding:3px;">h</th>
                            </tr>
                        </thead>
                        <tbody>`;
            for (let msg of sent) {{
                html += `<tr style="border-bottom:1px solid #ddd;">
                            <td style="padding:3px;">${{msg.t}}</td>
                            <td style="padding:3px;">Новый</td>
                            <td style="padding:3px;">${{msg.text.substring(0, 50)}}</td>
                            <td style="padding:3px;">${{msg.category}}</td>
                            <td style="padding:3px;">${{msg.h.toFixed(2)}}</td>
                          </tr>`;
            }}
            html += `</tbody>`;
        }}
    }} else {{
        let states = nodeStatesHistory[String(nodeId)];
        if (!states || states.length === 0) {{
            html = "<i>Нет истории состояний для этого узла</i>";
        }} else {{
            html = `<h4 style="margin:0 0 8px 0;">📈 UserState History (b, c, e)</h4>`;
            html += `<table style="width:100%; border-collapse:collapse; text-align:center; font-size:10px;">
                        <thead>
                            <tr style="background:#f0f0f0; border-bottom:1px solid #ccc;">
                                <th style="padding:3px;">t</th><th style="padding:3px;">b</th><th style="padding:3px;">c</th><th style="padding:3px;">e</th>
                            </tr>
                        </thead>
                        <tbody>`;
            for (let s of states) {{
                html += `<tr style="border-bottom:1px solid #ddd;">
                            <td style="padding:3px;">${{s.t}}</td>
                            <td style="padding:3px;">${{Number(s.b).toFixed(4)}}</td>
                            <td style="padding:3px;">${{Number(s.c).toFixed(4)}}</td>
                            <td style="padding:3px;">${{Number(s.e).toFixed(4)}}</td>
                          </tr>`;
            }}
            html += `</tbody>`;
        }}
    }}
    content.innerHTML = html;
    modal.style.display = "block";
}}

function showFullInfoEdge(edgeId) {{
    let info = edgeFullInfo[edgeId];
    let modal = document.getElementById("edgeModal");
    let content = document.getElementById("edgeModalContent");
    if (!modal) {{
        let modalHtml = `
        <div id="edgeModal" style="display:none; position:fixed; z-index:10000; left:0; top:0;
             width:100%; height:100%; background:rgba(0,0,0,0.6); backdrop-filter:blur(5px);">
            <div style="position:absolute; left:50%; top:50%; transform:translate(-50%,-50%);
                 background:white; border-radius:10px; width:60%; max-width:700px; max-height:70%;
                 box-shadow:0 5px 20px rgba(0,0,0,0.3); display:flex; flex-direction:column;">
                <div style="padding:10px 15px; border-bottom:1px solid #eee; display:flex; 
                     justify-content:space-between; align-items:center;">
                    <h3 style="margin:0; font-size:16px;">📋 Полная информация о ребре</h3>
                    <button onclick="closeModal('edgeModal')" style="background:none; border:none; font-size:22px; cursor:pointer;">&times;</button>
                </div>
                <div id="edgeModalContent" style="padding:12px; overflow-y:auto; flex:1; font-family:monospace; font-size:11px;"></div>
            </div>
        </div>`;
        document.body.insertAdjacentHTML('beforeend', modalHtml);
        modal = document.getElementById("edgeModal");
        content = document.getElementById("edgeModalContent");
    }}
    if (info && info.messages.length) {{
        content.innerHTML = `<b>🔷 EDGE ${{info.u}} → ${{info.v}}</b><br><hr>
            <b>📊 Всего сообщений:</b> ${{info.total}}<br><br>
            <div style="max-height:400px; overflow-y:auto;">${{info.messages.join('')}}</div>`;
    }} else {{
        content.innerHTML = "<i>На этом ребре нет сообщений</i>";
    }}
    modal.style.display = "block";
}}

function closeModal(modalId) {{
    let modal = document.getElementById(modalId);
    if(modal) modal.style.display = "none";
}}

function updateEdgeTooltip(edgeId, time) {{
    let edges = network.body.data.edges;
    let edge = edges.get(edgeId);
    if(!edge) return;
    let msgs = (edgeMessagesByTime[edgeId] || {{}})[time] || [];
    let lines = ["━━━━━━━━━━━━━━━━━━━━━━", "EDGE " + edge.from + " → " + edge.to, "⏰ ВРЕМЯ: " + time];
    if(msgs.length) {{
        lines.push("📨 СООБЩЕНИЙ: " + msgs.length);
        for(let i=0;i<msgs.length;i++){{
            lines.push("────────────────────");
            lines.push("📝: " + msgs[i].text.substring(0,50));
            lines.push("🏷️: " + msgs[i].category);
            lines.push("📊 h: " + msgs[i].h);
            if(msgs[i].age) lines.push("⏰ Age: " + msgs[i].age);
            lines.push("🤖 Blue Risk: " + msgs[i].blue_risk_score + " (" + msgs[i].blue_risk_level + ")");
        }}
    }} else {{
        lines.push("📭 Нет сообщений");
    }}
    lines.push("━━━━━━━━━━━━━━━━━━━━━━","💡 Двойной клик → полная информация");
    edges.update({{ id: edgeId, title: lines.join("\\n") }});
}}

function updateAllTooltips(time) {{
    for(let key in edgeMap) updateEdgeTooltip(edgeMap[key], time);
}}

function resetAllEdges() {{
    let edges = network.body.data.edges;
    let all = edges.get();
    for(let e of all) edges.update({{ id: e.id, color: "#95a5a6", width: 2 }});
}}

function highlightEdge(from, to, color) {{
    let eid = edgeMap[from+","+to];
    if(eid !== undefined) network.body.data.edges.update({{ id: eid, color: color, width: 4 }});
}}

function updateByTime(time) {{
    resetAllEdges();
    let events = timelineData[time] || [];
    for(let e of events){{
        let col = (e.category==="threat"||e.category==="manipulative") ? "#e74c3c" : "#3498db";
        highlightEdge(e.from, e.to, col);
    }}
    updateAllTooltips(time);
    document.getElementById("timeLabel").innerHTML = "⏰ TIME: " + time;
    document.getElementById("timeSlider").value = time;
}}

function playAnimation() {{
    if(animInterval){{
        clearInterval(animInterval); animInterval=null;
        document.getElementById("playBtn").innerHTML = "▶ Play";
        return;
    }}
    document.getElementById("playBtn").innerHTML = "⏸ Pause";
    animInterval = setInterval(() => {{
        if(currentTime >= {max_time}){{
            clearInterval(animInterval); animInterval=null;
            document.getElementById("playBtn").innerHTML = "▶ Play";
            return;
        }}
        currentTime++;
        updateByTime(currentTime);
    }}, 800);
}}

function resetAnimation() {{
    if(animInterval){{ clearInterval(animInterval); animInterval=null; }}
    currentTime=0; updateByTime(0);
    document.getElementById("playBtn").innerHTML = "▶ Play";
}}

function onTimeChange(val){{
    if(animInterval){{ clearInterval(animInterval); animInterval=null; document.getElementById("playBtn").innerHTML = "▶ Play"; }}
    currentTime = parseInt(val);
    updateByTime(currentTime);
}}

function toggleSettings(){{
    let panel = document.getElementById("settingsPanel");
    settingsVisible = !settingsVisible;
    panel.style.display = settingsVisible ? "block" : "none";
}}

function updateNodeSize(val){{
    let nodes = network.body.data.nodes;
    let allNodes = nodes.get();
    for(let n of allNodes){{
        nodes.update({{ id: n.id, size: Number(val), font: {{ size: Math.max(12, Number(val) * 0.35) }} }});
    }}
    network.redraw();
    document.getElementById("nodeSizeValue").innerHTML = val;
}}

function updateEdgeDistance(val){{
    network.setOptions({{ physics: {{ barnesHut: {{ springLength: parseInt(val) }} }} }});
    document.getElementById("edgeDistanceValue").innerHTML = val;
}}

network.once("stabilizationIterationsDone", function() {{
    updateByTime(0);
}});

network.on("doubleClick", function(params) {{
    if (params.nodes.length > 0) {{
        showFullInfoNode(params.nodes[0]);
    }} else if (params.edges.length > 0) {{
        showFullInfoEdge(params.edges[0]);
    }}
}});

document.addEventListener('click', function(e){{
    let em = document.getElementById('edgeModal');
    let nm = document.getElementById('nodeModal');
    if(e.target===em) em.style.display='none';
    if(e.target===nm) nm.style.display='none';
}});
</script>

<div style="position:fixed; bottom:20px; left:20px; z-index:999;">
  <button onclick="toggleSettings()" style="background:#34495e; color:white; border:none; padding:8px 15px; border-radius:6px; cursor:pointer; font-size:12px;">⚙ НАСТРОЙКИ</button>
  <div id="settingsPanel" style="display:none; position:fixed; bottom:80px; left:20px; background:white; padding:10px; border-radius:10px; box-shadow:0 4px 15px rgba(0,0,0,0.3); min-width:180px;">
    <div style="font-size:12px;"><div>📏 Размер узлов</div><input type="range" min="10" max="80" value="40" oninput="updateNodeSize(this.value)" style="width:100%;"><div>Текущий: <span id="nodeSizeValue">40</span> px</div></div>
    <div style="margin-top:8px; font-size:12px;"><div>📏 Дистанция рёбер</div><input type="range" min="80" max="800" value="260" oninput="updateEdgeDistance(this.value)" style="width:100%;"><div>Текущий: <span id="edgeDistanceValue">260</span> px</div></div>
  </div>
</div>

<div style="position:fixed; bottom:20px; right:20px; background:white; padding:8px 12px; border-radius:10px; box-shadow:0 0 12px rgba(0,0,0,0.2);">
  <div style="text-align:center; margin-bottom:6px; font-size:12px;"><b>📊 TIMELINE</b></div>
  <div style="display:flex; gap:8px; align-items:center;">
    <button id="playBtn" onclick="playAnimation()" style="background:#3498db; color:white; border:none; padding:5px 12px; border-radius:5px; cursor:pointer; font-size:11px;">▶ Play</button>
    <button onclick="resetAnimation()" style="background:#e74c3c; color:white; border:none; padding:5px 12px; border-radius:5px; cursor:pointer; font-size:11px;">⏮ Reset</button>
    <input type="range" id="timeSlider" min="0" max="{max_time}" value="0" onchange="onTimeChange(this.value)" style="width:280px;">
  </div>
  <div id="timeLabel" style="margin-top:5px; text-align:center; font-size:11px;">⏰ TIME: 0</div>
</div>

<style>
  .vis-tooltip {{ background: rgba(0,0,0,0.9); color: #fff; padding: 8px; border-radius: 6px; font-size: 10px; font-family: monospace; max-width: 450px; white-space: pre-line; z-index: 1000; }}
  ::-webkit-scrollbar {{ width: 6px; }}
  ::-webkit-scrollbar-track {{ background: #f1f1f1; border-radius: 3px; }}
  ::-webkit-scrollbar-thumb {{ background: #888; border-radius: 3px; }}
  table {{ font-size: 10px; }}
  th {{ padding: 3px; }}
  td {{ padding: 3px; }}
</style>
"""

    html = html.replace("</body>", custom_js + "</body>")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n✅ Visualization saved to: {output_path}")
    print(f"📊 Максимальное время: {max_time}")
    print(f"🔗 Всего рёбер: {edge_id}")
    return output_path