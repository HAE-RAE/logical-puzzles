"""Causal DAG (Directed Acyclic Graph) Reasoning Puzzle Generator

Generates causal reasoning puzzles based on event chains and time propagation.
Tests LLM's ability to reason about cause-effect relationships over time.

Key Features:
1. DAG-based event graph (no cycles)
2. Time-delayed causal relationships
3. Shortest path reasoning (Dijkstra)
4. Unique solution guarantee (deterministic graph)
"""

import random
import heapq
import json
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum


class EventType(Enum):
    """Event categories for puzzle generation"""
    TECHNICAL = "Technical"
    BUSINESS = "Business"
    ENVIRONMENTAL = "Environmental"
    OPERATIONAL = "Operational"


@dataclass
class CausalEdge:
    """Represents a causal relationship between events"""
    from_event: str
    to_event: str
    delay: int  # Time delay in minutes
    from_events: Optional[List[str]] = None  # Multiple prerequisites (for AND)
    condition: str = 'OR'  # 'OR' or 'AND'
    
    def __repr__(self):
        if self.from_events and len(self.from_events) > 1:
            cond = ' AND ' if self.condition == 'AND' else ' OR '
            return f"[{cond.join(self.from_events)}] → {self.to_event} (+{self.delay}min)"
        return f"{self.from_event} → {self.to_event} (+{self.delay}min)"


@dataclass
class Event:
    """Represents an event in the causal graph"""
    id: str
    name: str
    description: str
    event_type: EventType
    
    def __repr__(self):
        return f"{self.id}: {self.name}"


@dataclass
class CausalPuzzle:
    """Complete causal reasoning puzzle"""
    events: Dict[str, Event]
    edges: List[CausalEdge]
    trigger: str
    trigger_time: int
    target_event: str
    answer: int
    difficulty: str
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'events': {k: {
                'id': v.id,
                'name': v.name,
                'description': v.description,
                'event_type': v.event_type.value
            } for k, v in self.events.items()},
            'edges': [{
                'from_event': e.from_event,
                'to_event': e.to_event,
                'delay': e.delay,
                'from_events': e.from_events,
                'condition': e.condition
            } for e in self.edges],
            'trigger': self.trigger,
            'trigger_time': self.trigger_time,
            'target': self.target_event,
            'answer': self.answer,
            'difficulty': self.difficulty
        }


class CausalPuzzleGenerator:
    """Generate causal DAG reasoning puzzles"""
    
    def __init__(self):
        """Initialize with event templates"""
        self.event_templates = {
            EventType.TECHNICAL: [
                ('PowerOutage', 'Main power grid fails'),
                ('ServerDown', 'Application server becomes unavailable'),
                ('DatabaseCrash', 'Database service stops responding'),
                ('NetworkFailure', 'Network connectivity is lost'),
                ('DiskFull', 'Storage capacity reaches 100%'),
                ('MemoryLeak', 'Application memory usage spikes'),
                ('BackupFailed', 'Automated backup process fails'),
                ('SecurityBreach', 'Unauthorized access detected'),
                ('APITimeout', 'External API stops responding'),
                ('CacheExpired', 'Cache invalidation occurs'),
            ],
            EventType.BUSINESS: [
                ('OrderReceived', 'Customer places new order'),
                ('PaymentProcessed', 'Payment transaction completes'),
                ('InventoryLow', 'Stock level falls below threshold'),
                ('ShipmentDelayed', 'Delivery schedule is pushed back'),
                ('CustomerComplaint', 'Support ticket is created'),
                ('RefundIssued', 'Money is returned to customer'),
                ('PriceChanged', 'Product pricing is updated'),
                ('PromotionStarted', 'Marketing campaign launches'),
                ('ContractSigned', 'Legal agreement is finalized'),
                ('InvoiceSent', 'Billing document is generated'),
            ],
            EventType.ENVIRONMENTAL: [
                ('HeavyRain', 'Precipitation exceeds 50mm/hour'),
                ('RoadFlooded', 'Water level blocks traffic'),
                ('TrafficJam', 'Vehicle congestion occurs'),
                ('PowerSurge', 'Electrical grid experiences spike'),
                ('Earthquake', 'Seismic activity detected'),
                ('HeatWave', 'Temperature exceeds 35°C'),
                ('StormWarning', 'Severe weather alert issued'),
                ('Snowfall', 'Snow accumulation begins'),
                ('WindDamage', 'Strong winds cause infrastructure damage'),
                ('Drought', 'Water supply becomes limited'),
            ],
            EventType.OPERATIONAL: [
                ('MaintenanceScheduled', 'Planned system maintenance begins'),
                ('StaffShortage', 'Available workforce drops below minimum'),
                ('EquipmentFailure', 'Critical machinery stops working'),
                ('QualityIssue', 'Product defect is discovered'),
                ('SupplyChainDisruption', 'Vendor delivery is interrupted'),
                ('CapacityReached', 'Maximum throughput is exceeded'),
                ('PolicyChanged', 'New operational rules take effect'),
                ('InspectionFailed', 'Compliance check does not pass'),
                ('TrainingCompleted', 'Staff certification is achieved'),
                ('SystemUpgrade', 'Software version is updated'),
            ]
        }
    
    def generate_puzzle(self, difficulty: str, seed: Optional[int] = None) -> CausalPuzzle:
        """
        Generate a causal reasoning puzzle
        
        Args:
            difficulty: 'Easy', 'Medium', or 'Hard'
            seed: Random seed for reproducibility
        
        Returns:
            CausalPuzzle with unique solution
        """
        if seed is not None:
            random.seed(seed)
        
        # Difficulty configuration (calibrated for gpt-4o ~70/40/10%)
        config = {
            'Easy': {
                'num_events': random.randint(18, 22),
                'edge_density': 0.50,
                'delay_range': (15, 80),
                'max_out_degree': 3,
                'and_probability': 0.4,  # 40% AND conditions
            },
            'Medium': {
                'num_events': random.randint(45, 55),
                'edge_density': 0.70,
                'delay_range': (30, 150),
                'max_out_degree': 5,
                'and_probability': 0.92,  # 92% AND conditions
            },
            'Hard': {
                'num_events': random.randint(80, 100),
                'edge_density': 0.80,
                'delay_range': (50, 250),
                'max_out_degree': 7,
                'and_probability': 0.99,  # 99% AND conditions
            }
        }[difficulty]
        
        max_attempts = 100
        for attempt in range(max_attempts):
            try:
                # Generate events
                events = self._generate_events(config['num_events'])
                
                # Generate causal graph (DAG)
                edges = self._generate_causal_graph(events, config)
                
                if not edges:
                    continue
                
                # Select trigger (node with in-degree 0)
                in_degree = self._calculate_in_degree(events, edges)
                possible_triggers = [e_id for e_id, degree in in_degree.items() 
                                    if degree == 0]
                
                if not possible_triggers:
                    continue
                
                trigger = random.choice(possible_triggers)
                trigger_time = random.randint(0, 60)
                
                # Calculate reach times
                reach_times = self._calculate_reach_times(events, edges, trigger, trigger_time)
                
                # Select target (reachable event, not trigger)
                reachable = [e_id for e_id, time in reach_times.items() 
                           if time < float('inf') and e_id != trigger]
                
                if not reachable:
                    continue
                
                # Prefer events that are not too close or too far
                target_candidates = sorted(reachable, 
                                         key=lambda e: reach_times[e])
                
                # Pick from middle range for better difficulty
                mid_start = len(target_candidates) // 3
                mid_end = 2 * len(target_candidates) // 3
                if mid_end > mid_start:
                    target_event = random.choice(target_candidates[mid_start:mid_end])
                else:
                    target_event = random.choice(target_candidates)
                
                answer = reach_times[target_event]
                
                return CausalPuzzle(
                    events=events,
                    edges=edges,
                    trigger=trigger,
                    trigger_time=trigger_time,
                    target_event=target_event,
                    answer=answer,
                    difficulty=difficulty
                )
            
            except Exception as e:
                continue
        
        # If all attempts fail, generate simple linear chain
        return self._generate_simple_puzzle(difficulty)
    
    def _generate_events(self, num_events: int) -> Dict[str, Event]:
        """Generate event nodes"""
        events = {}
        
        # Distribute events across types
        event_types = list(EventType)
        selected = []
        
        for i in range(num_events):
            event_type = event_types[i % len(event_types)]
            available = [t for t in self.event_templates[event_type] 
                        if t[0] not in [s[0] for s in selected]]
            
            if not available:
                available = self.event_templates[event_type]
            
            name, description = random.choice(available)
            selected.append((name, description))
            
            events[f"E{i+1}"] = Event(
                id=f"E{i+1}",
                name=name,
                description=description,
                event_type=event_type
            )
        
        return events
    
    def _generate_causal_graph(self, events: Dict[str, Event], 
                               config: Dict) -> List[CausalEdge]:
        """Generate DAG of causal relationships with AND/OR conditions"""
        edges = []
        event_ids = sorted(events.keys())
        and_prob = config.get('and_probability', 0.0)
        
        # Create edges for each target node
        for i, to_id in enumerate(event_ids):
            if i == 0:
                continue  # Skip first node (trigger candidate)
            
            # Determine number of prerequisites
            max_prereqs = min(config['max_out_degree'], i)
            num_prereqs = random.randint(1, max(1, max_prereqs))
            
            # Select prerequisite events (from earlier nodes)
            possible_from = event_ids[:i]
            if not possible_from:
                continue
            
            from_events = random.sample(possible_from, min(num_prereqs, len(possible_from)))
            
            # Determine condition type
            condition = 'OR'
            if len(from_events) > 1 and random.random() < and_prob:
                condition = 'AND'
            
            delay = random.randint(*config['delay_range'])
            
            # Create edge
            if len(from_events) == 1:
                edges.append(CausalEdge(
                    from_event=from_events[0],
                    to_event=to_id,
                    delay=delay,
                    from_events=from_events,
                    condition='OR'
                ))
            else:
                edges.append(CausalEdge(
                    from_event=from_events[0],  # For compatibility
                    to_event=to_id,
                    delay=delay,
                    from_events=from_events,
                    condition=condition
                ))
        
        return edges
    
    def _calculate_in_degree(self, events: Dict[str, Event], 
                            edges: List[CausalEdge]) -> Dict[str, int]:
        """Calculate in-degree for each node"""
        in_degree = {e_id: 0 for e_id in events}
        for edge in edges:
            in_degree[edge.to_event] += 1
        return in_degree
    
    def _calculate_reach_times(self, events: Dict[str, Event],
                               edges: List[CausalEdge],
                               trigger: str,
                               trigger_time: int) -> Dict[str, int]:
        """
        Calculate earliest time each event occurs with AND/OR conditions
        
        Returns:
            Dictionary mapping event_id -> earliest occurrence time
        """
        # Build prerequisite tracking
        prereqs_for_event = {}
        edges_for_event = {}
        
        for edge in edges:
            from_events = edge.from_events if edge.from_events else [edge.from_event]
            prereqs_for_event[edge.to_event] = from_events
            edges_for_event[edge.to_event] = edge
        
        # Track earliest time each event occurs
        earliest_time = {e_id: float('inf') for e_id in events}
        earliest_time[trigger] = trigger_time
        
        # Track when each prerequisite reaches an event
        prereq_arrival_times = {e_id: {} for e_id in events}
        
        # Priority queue: (time, event_id)
        pq = [(trigger_time, trigger)]
        processed = set()
        
        while pq:
            current_time, current_event = heapq.heappop(pq)
            
            if current_event in processed:
                continue
            processed.add(current_event)
            
            # Find all events that depend on current_event
            for edge in edges:
                from_events = edge.from_events if edge.from_events else [edge.from_event]
                if current_event not in from_events:
                    continue
                
                to_event = edge.to_event
                arrival_time = current_time + edge.delay
                
                # Record this prerequisite's arrival time
                if current_event not in prereq_arrival_times[to_event]:
                    prereq_arrival_times[to_event][current_event] = arrival_time
                else:
                    # Keep earliest arrival from this prerequisite
                    prereq_arrival_times[to_event][current_event] = min(
                        prereq_arrival_times[to_event][current_event],
                        arrival_time
                    )
                
                # Check if all prerequisites have arrived
                all_prereqs_arrived = all(
                    prereq in prereq_arrival_times[to_event]
                    for prereq in from_events
                )
                
                if all_prereqs_arrived:
                    # Calculate trigger time based on condition
                    if edge.condition == 'AND':
                        # Wait for ALL prerequisites
                        trigger_time_for_event = max(
                            prereq_arrival_times[to_event][prereq]
                            for prereq in from_events
                        )
                    else:  # OR
                        # Trigger on FIRST prerequisite
                        trigger_time_for_event = min(
                            prereq_arrival_times[to_event][prereq]
                            for prereq in from_events
                        )
                    
                    # Update if this is earlier than current best
                    if trigger_time_for_event < earliest_time[to_event]:
                        earliest_time[to_event] = trigger_time_for_event
                        heapq.heappush(pq, (trigger_time_for_event, to_event))
        
        return earliest_time
    
    def _generate_simple_puzzle(self, difficulty: str) -> CausalPuzzle:
        """Generate simple linear chain as fallback"""
        num_events = 4
        events = self._generate_events(num_events)
        event_ids = sorted(events.keys())
        
        edges = []
        for i in range(len(event_ids) - 1):
            edges.append(CausalEdge(
                from_event=event_ids[i],
                to_event=event_ids[i+1],
                delay=random.randint(10, 30)
            ))
        
        trigger = event_ids[0]
        trigger_time = 0
        target_event = event_ids[-1]
        
        reach_times = self._calculate_reach_times(events, edges, trigger, trigger_time)
        answer = reach_times[target_event]
        
        return CausalPuzzle(
            events=events,
            edges=edges,
            trigger=trigger,
            trigger_time=trigger_time,
            target_event=target_event,
            answer=answer,
            difficulty=difficulty
        )
    
    def has_unique_solution(self, puzzle: CausalPuzzle) -> bool:
        """
        Verify puzzle has unique solution
        
        Since Dijkstra is deterministic, solution is always unique for a given graph.
        Just verify the graph is valid (DAG, connected).
        """
        # Check if target is reachable
        reach_times = self._calculate_reach_times(
            puzzle.events,
            puzzle.edges,
            puzzle.trigger,
            puzzle.trigger_time
        )
        
        return reach_times[puzzle.target_event] < float('inf')


def create_question(puzzle: CausalPuzzle) -> str:
    """Generate English question text for the puzzle"""
    
    # Format events
    event_lines = []
    for event_id in sorted(puzzle.events.keys()):
        event = puzzle.events[event_id]
        event_lines.append(f"  {event_id}: {event.name}")
        event_lines.append(f"      ({event.description})")
    
    events_description = '\n'.join(event_lines)
    
    # Format causal relationships
    causal_lines = []
    sorted_edges = sorted(puzzle.edges, key=lambda e: e.to_event)
    
    for edge in sorted_edges:
        from_events = edge.from_events if edge.from_events else [edge.from_event]
        to_name = puzzle.events[edge.to_event].name
        
        if len(from_events) == 1:
            from_name = puzzle.events[from_events[0]].name
            causal_lines.append(
                f"  {from_events[0]} ({from_name}) → "
                f"{edge.to_event} ({to_name}): {edge.delay} minutes"
            )
        else:
            if edge.condition == 'AND':
                prereq_str = ' AND '.join(f"{e} ({puzzle.events[e].name})" 
                                          for e in from_events)
                line = f"  [{prereq_str}] → {edge.to_event} ({to_name}): {edge.delay} minutes"
                line += "\n      (Requires ALL prerequisites)"
            else:
                prereq_str = ' OR '.join(f"{e} ({puzzle.events[e].name})" 
                                         for e in from_events)
                line = f"  [{prereq_str}] → {edge.to_event} ({to_name}): {edge.delay} minutes"
                line += "\n      (Triggered by FIRST prerequisite)"
            causal_lines.append(line)
    
    causality_description = '\n'.join(causal_lines)
    
    trigger_name = puzzle.events[puzzle.trigger].name
    target_name = puzzle.events[puzzle.target_event].name
    
    question = f"""You are analyzing a system of causal events and their propagation over time.

Events:
{events_description}

Causal Relationships (showing time delays):
{causality_description}

Rules:
- When an event occurs, it triggers its effects after the specified delay
- OR condition: Event occurs when the FIRST prerequisite reaches it
- AND condition: Event occurs only when ALL prerequisites have occurred
- All times are measured in minutes from a reference point (minute 0)

Initial Condition:
- Event {puzzle.trigger} ({trigger_name}) occurs at minute {puzzle.trigger_time}

Question:
At what minute does event {puzzle.target_event} ({target_name}) first occur?

Provide your answer as a single integer representing the minute number.
For example, if the event occurs 45 minutes after the start, answer: 45
"""
    
    return question


def generate_dataset(puzzles_per_difficulty: int = 3, verbose: bool = True) -> List[Dict]:
    """
    Generate a complete dataset of causal DAG puzzles
    
    Args:
        puzzles_per_difficulty: Number of puzzles per difficulty level
        verbose: Print generation progress
    
    Returns:
        List of puzzle dictionaries ready for evaluation
    """
    generator = CausalPuzzleGenerator()
    difficulties = ['Easy', 'Medium', 'Hard']
    dataset = []
    
    for difficulty in difficulties:
        if verbose:
            print(f"\n=== Generating {difficulty} puzzles ===")
        
        for i in range(puzzles_per_difficulty):
            puzzle = generator.generate_puzzle(difficulty)
            question = create_question(puzzle)
            
            puzzle_data = {
                'question': question,
                'answer': str(puzzle.answer),
                'difficulty': difficulty,
                'metadata': puzzle.to_dict()
            }
            
            dataset.append(puzzle_data)
            
            if verbose:
                print(f"  [{i+1}/{puzzles_per_difficulty}] "
                      f"{puzzle.trigger} → {puzzle.target_event}: "
                      f"{puzzle.answer} minutes")
    
    return dataset


def create_dataset_files(num_questions: int):
    """
    Create dataset files for causal DAG puzzles
    
    Args:
        num_questions: Number of questions to generate
        version: Version string for filenames
    
    Returns:
        Tuple[pd.DataFrame, List[Dict]]: (dataframe, json list)
    """
    import pandas as pd
    
    print(f"Generating {num_questions} causal DAG puzzles...")
    
    generator = CausalPuzzleGenerator()
    
    # Calculate puzzles per difficulty
    puzzles_per_diff = num_questions // 3
    remainder = num_questions % 3
    
    difficulties = ['Easy', 'Medium', 'Hard']
    all_puzzles = []
    
    for i, difficulty in enumerate(difficulties):
        count = puzzles_per_diff + (1 if i < remainder else 0)
        
        for j in range(count):
            puzzle = generator.generate_puzzle(difficulty, seed=i*1000+j)
            puzzle_data = {
                'question': create_question(puzzle),
                'answer': str(puzzle.answer),
                'difficulty': difficulty,
                'trigger': puzzle.trigger,
                'target': puzzle.target_event,
                'num_events': len(puzzle.events),
                'num_edges': len(puzzle.edges)
            }
            all_puzzles.append(puzzle_data)
    
    print(f"\nGenerated {len(all_puzzles)} puzzles")
    
    df = pd.DataFrame(all_puzzles)
    
    # Save files
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    
    # CSV
    csv_dir = PROJECT_ROOT / "data" / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    # Lowercase filename
    csv_path = csv_dir / "causal_dag.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"CSV file created: {csv_path}")
    
    # JSONL
    json_dir = PROJECT_ROOT / "data" / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = json_dir / "causal_dag.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item in all_puzzles:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"JSONL file created: {jsonl_path}")
    
    return df, all_puzzles


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Causal DAG Puzzle Generator")
    parser.add_argument("--num", type=int, default=300, help="Number of questions to generate")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Causal DAG Reasoning Puzzle Generator")
    print("=" * 70)
    
    create_dataset_files(num_questions=args.num)
    
    # Generate sample dataset
    # dataset = generate_dataset(puzzles_per_difficulty=2, verbose=True)
    
    # print("\n" + "=" * 70)
    # print("Sample Puzzle")
    # print("=" * 70)
    
    # sample = dataset[0]
    # print(sample['question'])
    # print(f"\n✅ Answer: {sample['answer']} minutes")
    
    # # Validate all puzzles
    # print("\n" + "=" * 70)
    # print("Validation")
    # print("=" * 70)
    
    # generator = CausalPuzzleGenerator()
    # for i, puzzle_data in enumerate(dataset):
    #     metadata = puzzle_data['metadata']
        
    #     # Reconstruct Event objects with proper EventType enum
    #     events = {}
    #     for k, v in metadata['events'].items():
    #         events[k] = Event(
    #             id=v['id'],
    #             name=v['name'],
    #             description=v['description'],
    #             event_type=EventType(v['event_type'])
    #         )
        
    #     puzzle = CausalPuzzle(
    #         events=events,
    #         edges=[CausalEdge(
    #             from_event=e['from_event'],
    #             to_event=e['to_event'],
    #             delay=e['delay'],
    #             from_events=e.get('from_events'),
    #             condition=e.get('condition', 'OR')
    #         ) for e in metadata['edges']],
    #         trigger=metadata['trigger'],
    #         trigger_time=metadata['trigger_time'],
    #         target_event=metadata['target'],
    #         answer=metadata['answer'],
    #         difficulty=metadata['difficulty']
    #     )
        
    #     is_valid = generator.has_unique_solution(puzzle)
    #     status = "✓" if is_valid else "✗"
    #     print(f"  Puzzle {i+1}: {status} {'Valid' if is_valid else 'Invalid'}")
    
    # print("\n✓ All puzzles generated successfully!")