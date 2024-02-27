import pandas as pd

filepath = 'test.csv'
header = {'Type': 0, 'sflow_agent_addr ess': 1, 'inputPort': 2, 'outputPort': 3, 'src_MAC': 4, 'dst_MAC': 5, 'ethernet_type': 6, 'in_vlan': 7, 'out_vlan': 8, 'src_IP': 9, 'dest_IP': 10, 'IP_protocol': 11,
          'ip_tos': 12, 'ip_ttl': 13, 'udp_src_port/tcp_ src_port/icmp_type': 14, 'udp_dst_port/tcp_ dst_port/icmp_code': 15, 'tcp_flags': 16, 'packet_size': 17, 'IP_size': 18, 'sampling_rate': 19}

df = pd.read_csv(filepath, names=['Type', 'sflow_agent_addr ess', 'inputPort', 'outputPort', 'src_MAC', 'dst_MAC', 'ethernet_type', 'in_vlan', 'out_vlan', 'src_IP', 'dest_IP',
                 'IP_protocol', 'ip_tos', 'ip_ttl', 'udp_src_port/tcp_src_port/icmp_type', 'udp_dst_port/tcp_ dst_port/icmp_code', 'tcp_flags', 'packet_size', 'IP_size', 'sampling_rate'])
# Top 5 Talkers
top5Talkers = df['src_IP'].value_counts().index.tolist()[:5]
top5TalkersNumberOfPackets = df['src_IP'].value_counts().values.tolist()[:5]

# Top 5 Listeners
top5Listeners = df['dst_IP'].value_counts().index.tolist()[:5]
top5ListenersNumberOfPackets = df['dst_IP'].value_counts().values.tolist()[:5]

# Top 5 applications
top5Apps = df['udp_dst_port/tcp_dst_port/icmp_code'].value_counts().index.tolist()[:5]
top5AppsNumberOfPackets = df['udp_dst_port/tcp_dst_port/icmp_code'].value_counts(
).values.tolist()[:5]

# Total traffic
totalTraffic = df.sum(axis=18, skipna=True)

# Proportion of TCP and UDP packets
df[df.last == 'smith'].shape[0]
totalPackets = len(df.index)
TCPPackets = df[df.IP_protocol == 6].shape[0]
UDPPackets = df[df.IP_protocol == 17].shape[0]
TCPPortion = TCPPackets / totalPackets
UDPPortion = UDPPackets / totalPackets

# Top 5 communication pair
df['comminication_pair'] = df['src_IP'] + ' ' + df['dst_IP']
top5ComunicationPairs = df['comminication_pair'].value_counts().index.tolist()[
    :5]
top5ComunicationPairsNumberOfCom = df['comminication_pair'].value_counts().values.tolist()[
    :5]
