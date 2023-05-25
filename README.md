# Deep-State-Machine: Generative Model for Graph Generation
	- Deep Auto-Regressive Machines for Graph Construction
 
# Deep State Machine (DSM): Envisioned a novel appraoch for learning and generating graphs.
	- Multiple states
	- Complex states
	- Customizable states
	- Graph Deconstruction Tree Model (GDTM) for construction sequence generation
	- GDTM: A non-deterministic approach for traversing graph
	- Unparameterized construction sequences
	- Learning complex embeddings in single state
	- Learn and generate complex structures of graph in single state
	- Graph generation in fewer steps
	- DSM combined with GDTM to learn various alternative paths of graph generation


![Deep State Machine](Deep-State-Machine.jpg)
	
# Graph Deconstruction Tree Model: 
	- Non-deterministic approach of graph traversal
	- Navigates several alternative paths through which graphs can be generated
	- Generates construction sequence through deconstruction & construction method
	- Policy of randomly transitioning between simple to complex valid decision operations
	
<div align="center">
	<img src="gdtm.jpg" alt="gdtm" width="720" height="700">
</div>

# DGMG, DeepGG, GraphRNN: Previous Research
 	- Basic two states, [add node, add edge]
 	- Simple and non-customizable states
 	- Traversal algorithm: bfs, dfs,...
 	- Parameterized construction sequencess
 	- Learning embeddings of either node or edge in a single state

