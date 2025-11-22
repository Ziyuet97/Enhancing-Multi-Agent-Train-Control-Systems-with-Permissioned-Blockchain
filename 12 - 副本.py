import random
import csv  # Added for saving results

NUM_SERVERS = 30

RPC_TIMEOUT = 500
MIN_RPC_LATENCY = 10
MAX_RPC_LATENCY = 50
ELECTION_TIMEOUT = 1000
MAXP1 = 30
MAXP2 = 50
MAXP3 = 20

# This will be modified in the main loop
RANDOM_SECTION = 12 

BATCH_SIZE = 2

class model:
    def __init__(self, time, messages):
        self.time = time
        self.messages = messages

class server:
    def __init__(self, ID, state, term, commitIndex, matchIndex, peers, votedFor, log, electionAlarm, voteGranted, p1, p2, p3, nextIndex, rpcDue, heartbeatDue):
        self.ID = ID
        self.state = state
        self.term = term
        self.commitIndex = commitIndex
        self.matchIndex = matchIndex
        self.peers = peers
        self.votedFor = votedFor
        self.log = log
        self.electionAlarm = electionAlarm
        self.voteGranted = voteGranted
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.nextIndex = nextIndex
        self.rpcDue = rpcDue
        self.heartbeatDue = heartbeatDue

class message:
    def __init__(self, to, come, category, term, lastLogTerm, lastLogIndex, direction, sendTime, recvTime):
        self.to = to
        self.come = come
        self.category = category
        self.term = term
        self.lastLogTerm = lastLogTerm
        self.lastLogIndex = lastLogIndex
        self.direction = direction
        self.sendTime = sendTime
        self.recvTime = recvTime

class request:
    def __init__(self, to, come, category, term, lastLogTerm, lastLogIndex, direction, sendTime, recvTime, entries):
        self.to = to
        self.come = come
        self.category = category
        self.term = term
        self.lastLogTerm = lastLogTerm
        self.lastLogIndex = lastLogIndex
        self.direction = direction
        self.sendTime = sendTime
        self.recvTime = recvTime
        self.entries = entries

class reply:
    def __init__(self, to, come, category, term, lastLogTerm, lastLogIndex, direction, granted, sendTime, recvTime, success, matchIndex):
        self.to = to
        self.come = come
        self.category = category
        self.term = term
        self.lastLogTerm = lastLogTerm
        self.lastLogIndex = lastLogIndex
        self.direction = direction
        self.granted = granted
        self.sendTime = sendTime
        self.recvTime = recvTime
        self.success = success
        self.matchIndex = matchIndex
        
class entries:
    def __init__(self, prevIndex, prevTerm, logSlot, lastIndex):
        self.prevIndex = prevIndex
        self.prevTerm = prevTerm
        self.logSlot = logSlot
        self.lastIndex = lastIndex

def makeMap(keys, value):
    map = {}
    for k in keys:
        map[k] = value
    return map

# from JS

def sendMessage(model, message):
    message.sendTime = model.time
    message.recvTime = model.time + random.randint(MIN_RPC_LATENCY, MAX_RPC_LATENCY)
    push(model, message)

def sendRequest(model, request):
    request.direction = 'request'
    sendMessage(model, request)

def sendReply(model, request, reply):
    reply.come = request.to
    reply.to = request.come
    reply.category = request.category
    reply.direction = 'reply'
    sendMessage(model, reply)

# return term of NO.index
def logTerm(log, index):
    if (index < 1 or index > len(log)):
        return 0
    else:
        return log[index-1][0]

def makeElectionAlarm(now, server):
    # Note: This uses the global RANDOM_SECTION which is updated in the main loop
    if server.p1 + server.p2 + server.p3 > RANDOM_SECTION:
        MIN = (server.p1 + server.p2 + server.p3 - RANDOM_SECTION) / 100.0 + 1
    else:
        MIN = 1 
    return now + random.uniform(MIN * ELECTION_TIMEOUT, ((server.p1 + server.p2 + server.p3) / 100.0 + 1) * ELECTION_TIMEOUT)

# give a node a role of follower and renew the alarm and term by leader's
def stepDown(model, server, term):
    server.term = term
    server.state = 'follower'
    server.votedFor = None
    if server.electionAlarm <= model.time:
        server.electionAlarm = makeElectionAlarm(model.time, server)
        
def startNewElection(model, server):
    if ((server.state == 'follower' or server.state == 'candidate') and server.electionAlarm <= model.time):
        server.electionAlarm = makeElectionAlarm(model.time, server)
        server.term += 1
        server.votedFor = server.ID
        server.state = 'candidate'
        server.voteGranted  = makeMap(server.peers, False)
        server.matchIndex   = makeMap(server.peers, 0)
        server.nextIndex    = makeMap(server.peers, 1)
        server.rpcDue       = makeMap(server.peers, 0)
        server.heartbeatDue = makeMap(server.peers, 0)
        # print(server.ID, 'start election of term', server.term)

def sendRequestVote(model, server, peer):
    if (server.state == 'candidate' and server.rpcDue[peer.ID] <= model.time):
        server.rpcDue[peer.ID] = model.time + RPC_TIMEOUT
        M = message(peer, server, 'RequestVote', server.term, logTerm(server.log, len(server.log)), len(server.log), None, None, None)
        sendRequest(model, M)    
            
def becomeLeader(model, server):
    i = 0
    for value in server.voteGranted.values():
        if value == True:
            i += 1
    if (server.state == 'candidate' and i + 1 > int(NUM_SERVERS / 2)):
        # print('server ' + server.ID + ' is leader in term ' + str(server.term)+ ' , current time is '+ str(model.time), server.p1, server.p2, server.p3)

        server.state = 'leader'
        server.nextIndex = makeMap(server.peers, len(server.log) + 1)
        server.rpcDue = makeMap(server.peers, float('inf'))
        server.heartbeatDue = makeMap(server.peers, 0)
        server.electionAlarm = float('inf')
        return model.time
    return None

# log synchronize
def sendAppendEntries(model, server, peer):
    if (server.state == 'leader' and (server.heartbeatDue[peer.ID] <= model.time or (server.nextIndex[peer.ID] <= len(server.log) and server.rpcDue[peer.ID] <= model.time))):
        prevIndex = server.nextIndex[peer.ID] - 1
        lastIndex = min(prevIndex + BATCH_SIZE, len(server.log))
        if (server.matchIndex[peer.ID] + 1 < server.nextIndex[peer.ID]):
            lastIndex = prevIndex
        E = entries(prevIndex, logTerm(server.log, prevIndex), server.log[prevIndex: lastIndex], min(server.commitIndex, lastIndex))
        R = request(peer, server, 'AppendEntries', server.term, None, None, None, None, None, E)
        sendRequest(model, R)
        server.rpcDue[peer.ID] = model.time + RPC_TIMEOUT
        server.heartbeatDue[peer.ID] = model.time + ELECTION_TIMEOUT / 2

def handleRequestVoteRequest(model, server, request):
    if (server.term < request.term):
        stepDown(model, server, request.term)
    granted = False

    if (server.term == request.term and (server.votedFor == None or server.votedFor == request.come.ID)):
        # Raft 安全性检查：比较日志新旧
        logOk = False
        myLastLogTerm = logTerm(server.log, len(server.log))
        if request.lastLogTerm > myLastLogTerm:
            logOk = True
        elif request.lastLogTerm == myLastLogTerm and request.lastLogIndex >= len(server.log):
            logOk = True
            
        if logOk:
            granted = True
            server.votedFor = request.come.ID
            server.electionAlarm = makeElectionAlarm(model.time, server)
        
    R = reply(request.to, request.come, None, server.term, None, None, None, granted, None, None, None, None)
    sendReply(model, request, R)

def handleRequestVoteReply(model, server, reply):
    if (server.term < reply.term):
        stepDown(model, server, reply.term)

    if (server.state == 'candidate' and server.term == reply.term):
        server.rpcDue[reply.come.ID] = float('inf')
        server.voteGranted[reply.come.ID] = reply.granted

def handleAppendEntriesRequest(model, server, request):
    success = False
    matchIndex = 0
    if (server.term < request.term):
        stepDown(model, server, request.term)
    if (server.term == request.term):
        server.state = 'follower'
        server.electionAlarm = makeElectionAlarm(model.time, server)
        if (request.entries.prevIndex == 0 or (request.entries.prevIndex <= len(server.log) and logTerm(server.log, request.entries.prevIndex) == request.entries.prevTerm)):
            success = True
            index = request.entries.prevIndex
            for i in range(0, len(request.entries.logSlot)):
                index += 1
                if index > len(server.log):
                    server.log.append(request.entries.logSlot[i])
                elif (logTerm(server.log, index) != request.entries.logSlot[i][0]):
                    del server.log[index-1:]
                    server.log.append(request.entries.logSlot[i])
                    
            matchIndex = index
            server.commitIndex = max(server.commitIndex, request.entries.lastIndex)
    R = reply(request.to, request.come, None, server.term, None, None, None, None, None, None, success, matchIndex)
    sendReply(model, request, R)  
    
def handleAppendEntriesReply(model, server, reply):
    if (server.term < reply.term):
        stepDown(model, server, reply.term)
    if (server.state == 'leader' and server.term == reply.term):
        if (reply.success):
            server.matchIndex[reply.come.ID] = max(server.matchIndex[reply.come.ID], reply.matchIndex)
            server.nextIndex[reply.come.ID] = reply.matchIndex + 1
        else:
            server.nextIndex[reply.come.ID] = max(1, server.nextIndex[reply.come.ID] - 1)
    server.rpcDue[reply.come.ID] = 0

def push(model, message):
    model.messages.append(message)

# --- 优化后的 deliver：O(N) 复杂度，非破坏性遍历 ---
def deliver(model):
    if not model.messages:
        return

    arrived = []
    pending = []
    
    # 分离已到达和未到达的消息
    for msg in model.messages:
        if msg.recvTime <= model.time:
            arrived.append(msg)
        else:
            pending.append(msg)
    
    model.messages = pending

    for message in arrived:
        server = message.to
        
        # 提取 lastLogTerm，防止空日志崩溃
        sender_log_term = 0
        sender_log_len = 0
        
        # Reply 类型消息可能不关注 come 的 log，或者 come 属性不存在（取决于实现细节，这里安全起见做检查）
        if hasattr(message, 'come') and hasattr(message.come, 'log'):
            sender_log = message.come.log
            sender_log_len = len(sender_log)
            if sender_log_len > 0:
                sender_log_term = sender_log[-1][0]

        if message.category == 'RequestVote':
            if message.direction == 'request':
                request = message 
                request.lastLogTerm = sender_log_term
                request.lastLogIndex = sender_log_len
                handleRequestVoteRequest(model, server, request)
            elif message.direction == 'reply':
                handleRequestVoteReply(model, server, message)
                
        elif message.category == 'AppendEntries':
            if message.direction == 'request':
                handleAppendEntriesRequest(model, server, message)
            elif message.direction == 'reply':
                handleAppendEntriesReply(model, server, message)

def initialize(nodes):
    peers = []
    for i in range(NUM_SERVERS):
        peers.append(f'node{i}')

    for i in range(NUM_SERVERS):
        ID = f'node{i}'
        
        log = list()
        log.append([1, 'apple'])
        log.append([1, 'banana'])
        log.append([2, 'cat'])
        log.append([2, 'dog'])
        
        matchIndex = {}
        voteGranted = {}
        nextIndex = {}
        rpcDue = {}
        heartbeatDue = {}
        
        for p in peers:
            matchIndex[p] = 4
            voteGranted[p] = False
            nextIndex[p] = 5
            rpcDue[p] = 0
            heartbeatDue[p] = 0
        
        voteFor = None
        
        p1 = random.random() * MAXP1
        p2 = random.random() * MAXP2
        p3 = random.random() * MAXP3
        election = 0
        
        nodes.append(server(str(ID), 'follower', 2, 5, matchIndex, peers, voteFor, log, election, voteGranted, p1, p2, p3, nextIndex, rpcDue, heartbeatDue))

    return nodes

# --- 优化后的 update：移除内部 deliver，减少循环次数 ---
def update(model, nodes):
    # 1. 统一处理消息分发
    deliver(model)
    
    leader_found_time = None

    # 2. 处理状态转换
    for node in nodes:
        startNewElection(model, node)
        RE = becomeLeader(model, node)
        if RE is not None:
            leader_found_time = RE
    
    # 3. 发送 RPC (仅 Leader 和 Candidate 需要主动发送)
    for i in range(len(nodes)):
        server_node = nodes[i]
        
        # 性能优化：Follower 不主动发送 RPC
        if server_node.state == 'follower':
            continue
            
        for j in range(len(nodes)):
            if i == j: continue
            peer_node = nodes[j]
            
            if server_node.state == 'candidate':
                sendRequestVote(model, server_node, peer_node)
            
            elif server_node.state == 'leader':
                sendAppendEntries(model, server_node, peer_node)
                
    return leader_found_time

# --- 新增：计算下一个事件时间，实现时间跳跃 ---
def get_next_event_time(model, nodes):
    min_next_time = float('inf')
    
    # 1. 下一条消息到达时间
    for msg in model.messages:
        if msg.recvTime < min_next_time:
            min_next_time = msg.recvTime
            
    # 2. 节点超时时间 (Election 或 RPC)
    for node in nodes:
        if node.electionAlarm < min_next_time:
            min_next_time = node.electionAlarm
            
        if node.state == 'leader' or node.state == 'candidate':
            # 检查 RPC 重试和心跳
            for t in node.rpcDue.values():
                if t > model.time and t < min_next_time:
                    min_next_time = t
            for t in node.heartbeatDue.values():
                 if t > model.time and t < min_next_time:
                    min_next_time = t
                    
    return min_next_time

def stop(model, server):
    server.state = 'stopped'
    server.electionAlarm = 0

def resume(model, server):
    server.state = 'follower'
    server.electionAlarm = makeElectionAlarm(model.time)

if __name__ == '__main__':
    
    output_filename = "raft_experiment_results.csv"
    
    print(f"Starting Simulation. Results will be saved to {output_filename}")
    print("-" * 60)
    print(f"{'R_SEC':<6} | {'Avg P1':<8} | {'Avg P2':<8} | {'Avg P3':<8} | {'Sum P':<8} | {'Time':<8}")
    print("-" * 60)

    with open(output_filename, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # Write Header
        writer.writerow(['RANDOM_SECTION', 'Avg_P1', 'Avg_P2', 'Avg_P3', 'Avg_Sum_P', 'Avg_Election_Time', 'Term_Distribution_ACMLT'])

        # --- Outer Loop: Iterate RANDOM_SECTION from 0 to 25 ---
        for section_val in range(51):
            RANDOM_SECTION = section_val
            
            # Reset Statistics for this section
            ACMLT = []
            for i in range(20): ACMLT.append(0)
            
            P1_stats = []
            P2_stats = []
            P3_stats = []
            Sum_stats = []
            Timescar = []
            
            TOTAL_ROUNDS = 2000
            
            # Resize stat lists
            for i in range(TOTAL_ROUNDS):
                P1_stats.append(0)
                P2_stats.append(0)
                P3_stats.append(0)
                Sum_stats.append(0)
                Timescar.append(0)
            
            # Reset Model
            sim_model = model(0, [])

            # --- Inner Loop: Run Rounds ---
            for x in range(TOTAL_ROUNDS):  
                sim_model.time = 1000
                sim_model.messages = [] # Clear messages
                nodes = []
                
                nodes = initialize(nodes)
                
                for node in nodes:
                    node.electionAlarm = makeElectionAlarm(sim_model.time, node)
                    
                end_time = sim_model.time + 15000
                
                # Simulation Time Loop
                while sim_model.time < end_time:
                    RE = update(sim_model, nodes)
                    
                    if RE is not None:
                        Timescar[x] = RE
                        # Optional: break if we only care about first leader election time
                        # break 
                    
                    next_event = get_next_event_time(sim_model, nodes)
                    
                    if next_event == float('inf') or next_event > end_time:
                        sim_model.time = end_time
                    else:
                        sim_model.time = max(sim_model.time + 1, next_event)
                
                # Collect Round Stats
                for node in nodes:
                    if node.state == 'leader':
                        while node.term >= len(ACMLT):
                            ACMLT.append(0)
                            
                        ACMLT[node.term] += 1
                        
                        P1_stats[x] = node.p1
                        P2_stats[x] = node.p2
                        P3_stats[x] = node.p3
                        Sum_stats[x] = node.p1 + node.p2 + node.p3
            
            # --- Calculate Averages ---
            def safe_avg(lst):
                valid = [v for v in lst if v != 0] 
                return sum(valid) / len(valid) if len(valid) > 0 else 0

            avg_p1 = safe_avg(P1_stats)
            avg_p2 = safe_avg(P2_stats)
            avg_p3 = safe_avg(P3_stats)
            avg_sum = safe_avg(Sum_stats)
            avg_time = safe_avg(Timescar)
            
            # --- Print to Console ---
            print(f"{RANDOM_SECTION:<6} | {avg_p1:<8.2f} | {avg_p2:<8.2f} | {avg_p3:<8.2f} | {avg_sum:<8.2f} | {avg_time:<8.2f}")

            # --- Save to File ---
            writer.writerow([RANDOM_SECTION, avg_p1, avg_p2, avg_p3, avg_sum, avg_time, ACMLT])

    print("-" * 60)
    print("Simulation complete. Data saved.")