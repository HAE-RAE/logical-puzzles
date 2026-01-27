import random
from pathlib import Path
import functools
import logging
import json
import pandas as pd

@functools.lru_cache(None)
def get_relation_chain_to_title():
    return {
        # ("나", "아버지"): "아버지",
        # ("나", "어머니"): "어머니",
        
        # === 친가 (아버지 쪽) ===
        ("나", "아버지", "아버지"): "친할아버지",
        ("나", "아버지", "어머니"): "친할머니",
        ("나", "아버지", "형"): ["큰아버지", "백부"],
        ("나", "아버지", "형", "아내"): ["큰어머니", "백모"],
        ("나", "아버지", "형", "자녀"): ["종형제", "사촌"],
        ("나", "아버지", "남동생"): ["작은아버지", "숙부", "삼촌"],
        ("나", "아버지", "남동생", "아내"): ["작은어머니", "숙모"],
        ("나", "아버지", "남동생", "자녀"): ["종형제", "사촌"],
        ("나", "아버지", "누나"): "고모",
        ("나", "아버지", "누나", "남편"): "고모부",
        ("나", "아버지", "누나", "자녀"): ["내종형제", "고종사촌"],
        ("나", "아버지", "여동생"): "고모",
        ("나", "아버지", "여동생", "남편"): "고모부",
        ("나", "아버지", "여동생", "자녀"): ["내종형제", "고종사촌"],

        # === 증조/방계 ===
        ("나", "아버지", "아버지", "아버지"): "증조할아버지",
        ("나", "아버지", "아버지", "어머니"): "증조할머니",
        ("나", "아버지", "아버지", "형"): ["큰할아버지", "백조부"],
        ("나", "아버지", "아버지", "형", "아내"): ["큰할머니", "백조모"],
        ("나", "아버지", "아버지", "남동생"): ["작은할아버지", "숙조부"],
        ("나", "아버지", "아버지", "남동생", "아내"): ["작은할머니", "숙조모"],
        ("나", "아버지", "아버지", "누나"): ["고모할머니", "대고모"],
        ("나", "아버지", "아버지", "누나", "남편"): ["고모할아버지", "대고모부"],
        ("나", "아버지", "아버지", "여동생"): ["고모할머니", "대고모"],
        ("나", "아버지", "아버지", "여동생", "남편"): ["고모할아버지", "대고모부"],

        # === 진외가 ===
        ("나", "아버지", "어머니", "오빠"): ["진외종조부", "진외할아버지"],
        ("나", "아버지", "어머니", "남동생"): ["진외종조부", "진외할아버지"],
        ("나", "아버지", "어머니", "언니"): ["진외이모할머니", "대이모"],
        ("나", "아버지", "어머니", "언니", "남편"): ["진외이모할아버지", "대이모부"],
        ("나", "아버지", "어머니", "여동생"): ["진외이모할머니", "대이모"],
        ("나", "아버지", "어머니", "여동생", "남편"): ["진외이모할아버지", "대이모부"],

        # === 외가 (어머니 쪽) ===
        ("나", "어머니", "아버지"): ["외할아버지", "외조부"],
        ("나", "어머니", "어머니"): ["외할머니", "외조모"],
        ("나", "어머니", "오빠"): ["큰외삼촌", "외숙부", "외삼촌"],
        ("나", "어머니", "오빠", "아내"): ["큰외숙모", "외숙모"],
        ("나", "어머니", "오빠", "자녀"): ["외사촌", "외종형제"],
        ("나", "어머니", "남동생"): ["작은외삼촌", "외숙부", "외삼촌"],
        ("나", "어머니", "남동생", "아내"): ["작은외숙모", "외숙모"],
        ("나", "어머니", "남동생", "자녀"): ["외사촌", "외종형제"],
        ("나", "어머니", "언니"): ["큰이모", "이모"],
        ("나", "어머니", "언니", "남편"): ["큰이모부", "이모부"],
        ("나", "어머니", "언니", "자녀"): ["이종사촌", "이종형제"],
        ("나", "어머니", "여동생"): ["작은이모", "이모"],
        ("나", "어머니", "여동생", "남편"): ["작은이모부", "이모부"],
        ("나", "어머니", "여동생", "자녀"): ["이종사촌", "이종형제"],

        # === 외증조/외방계 ===
        ("나", "어머니", "아버지", "아버지"): ["증조외할아버지", "외증조부"],
        ("나", "어머니", "아버지", "어머니"): ["증조외할머니", "외증조모"],
        ("나", "어머니", "아버지", "형"): ["큰외할아버지", "외종조부"],
        ("나", "어머니", "아버지", "형", "아내"): ["큰외할머니", "외종조모"],
        ("나", "어머니", "아버지", "남동생"): ["작은외할아버지", "외종조부"],
        ("나", "어머니", "아버지", "남동생", "아내"): ["작은외할머니", "외종조모"],
        ("나", "어머니", "아버지", "누나"): ["고모외할머니", "외대고모"],
        ("나", "어머니", "아버지", "누나", "남편"): ["고모외할아버지", "외대고모부"],
        ("나", "어머니", "아버지", "여동생"): ["고모외할머니", "외대고모"],
        ("나", "어머니", "아버지", "여동생", "남편"): ["고모외할아버지", "외대고모부"],

        # === 외외가 ===
        ("나", "어머니", "어머니", "오빠"): ["외삼촌할아버지", "외종조부"], 
        ("나", "어머니", "어머니", "남동생"): ["외삼촌할아버지", "외종조부"],
        ("나", "어머니", "어머니", "언니"): ["큰이모할머니", "이모할머니", "대이모"],
        ("나", "어머니", "어머니", "언니", "남편"): ["이모할아버지", "대이모부"],
        ("나", "어머니", "어머니", "여동생"): ["작은이모할머니", "이모할머니", "대이모"],
        ("나", "어머니", "어머니", "여동생", "남편"): ["이모할아버지", "대이모부"],

        # === 배우자 관계 ===
        ("나", "남편", "아버지"): ["시아버지", "아버님"],
        ("나", "남편", "어머니"): ["시어머니", "어머님"],
        ("나", "남편", "형"): ["아주버님", "시숙"],
        ("나", "남편", "형", "아내"): "형님",
        ("나", "남편", "남동생"): ["도련님", "서방님"], 
        ("나", "남편", "남동생", "아내"): "동서",
        ("나", "남편", "누나"): ["형님", "시누이"],
        ("나", "남편", "누나", "남편"): ["아주버님", "고모부님"],
        ("나", "남편", "여동생"): ["아가씨", "시누이"],
        ("나", "남편", "여동생", "남편"): "서방님",

        ("나", "아내", "아버지"): ["장인", "장인어른", "아버님"],
        ("나", "아내", "어머니"): ["장모", "장모님", "어머님"],
        ("나", "아내", "오빠"): ["형님", "처남"], 
        ("나", "아내", "오빠", "아내"): ["아주머니", "처남댁"],
        ("나", "아내", "남동생"): "처남",
        ("나", "아내", "남동생", "아내"): "처남댁",
        ("나", "아내", "언니"): "처형",
        ("나", "아내", "언니", "남편"): ["형님", "동서"], 
        ("나", "아내", "여동생"): "처제",
        ("나", "아내", "여동생", "남편"): ["동서", "제부", "서방"],


        ("나", "아버지", "형"): ["큰아버지", "백부"],
        ("나", "아버지", "형", "아내"): ["큰어머니", "백모"],
        ("나", "아버지", "형", "자녀"): ["종형제", "사촌"],
        ("나", "아버지", "남동생"): ["작은아버지", "숙부", "삼촌"],
        ("나", "아버지", "남동생", "아내"): ["작은어머니", "숙모"],
        ("나", "아버지", "남동생", "자녀"): ["종형제", "사촌"],
        ("나", "아버지", "누나"): "고모",
        ("나", "아버지", "누나", "남편"): "고모부",
        ("나", "아버지", "누나", "자녀"): ["내종형제", "고종사촌"],
        ("나", "아버지", "여동생"): "고모",
        ("나", "아버지", "여동생", "남편"): "고모부",
        ("나", "아버지", "여동생", "자녀"): ["내종형제", "고종사촌"],

        # ("나", "형", "자녀"): ["조카", "질자", "질녀"],
        # ("나", "남동생", "자녀"): ["조카", "질자", "질녀"],
        # ("나", "누나", "자녀"): ["조카", "생질", "고종조카"],
        # ("나", "여동생", "자녀"): ["조카", "생질", "고종조카"],

    }

@functools.lru_cache(None)
def get_friend_names():
    return ["준용", "규진", "한울", "다솔", "승혁", "영숙", "상대", "대곤", "수용", "현우"]

@functools.lru_cache(None)
def get_person_descriptors():
    return [
        "파란 셔츠 입은 분", "빨간 옷 입은 분", "흰 티셔츠 입은 분", 
        "검은 정장 입은 분", "회색 자켓 입은 분", "노란 니트 입은 분",
        "초록 스웨터 입은 분", "보라색 옷 입은 분", "갈색 재킷 입은 분",
        "안경 쓴 분", "긴 머리 분", "짧은 머리 분", 
        "선글라스 쓰신 분", "모자 쓴 분", "환하게 웃고 있는 분",
        "왼쪽에 계신 분", "오른쪽에 계신 분", "가운데 계신 분",
        "맨 앞에 계신 분", "뒤에 계신 분", "옆에 계신 분"
    ]

@functools.lru_cache(None)
def get_dialogue_templates():
    """
    Returns natural dialogue templates (questions and answers) for each relationship.
    Key: Tuple of (subject of the relationship, target)
    Value: List of dialogue strings (in order: question, answer)
    {speaker}: The person asking the question
    {source}: The person from whom the relationship is referenced
    {target}: The person being asked about
    {particle}: Appropriate grammatical particle (e.g., topic marker)
    """
    return {
        # === 기본 관계 질문 (나 -> 부모님/배우자) ===
        ('나', '아버지'): [
            "{speaker}: \"이 사진 속 {target}{particle} 누구야?\"",
            "나: \"아, 그분은 우리 아버지야.\""
        ],
        ('나', '어머니'): [
            "{speaker}: \"이 사진 속 {target}{particle} 누구야?\"",
            "나: \"아, 그분은 우리 어머니야.\""
        ],
        
        # === 형제자매 관계 질문 ===
        ('나', '형'): [
            "{speaker}: \"이 사진 속 {target}{particle} 누구야?\"",
            "나: \"아, 그분은 내 형이야.\""
        ],
        ('나', '누나'): [
            "{speaker}: \"이 사진 속 {target}{particle} 누구야?\"",
            "나: \"아, 그분은 내 누나야.\""
        ],
        ('나', '남동생'): [
            "{speaker}: \"이 사진 속 {target}{particle} 누구야?\"",
            "나: \"아, 그분은 내 남동생이야.\""
        ],
        ('나', '여동생'): [
            "{speaker}: \"이 사진 속 {target}{particle} 누구야?\"",
            "나: \"아, 그분은 내 여동생이야.\""
        ],
        
        # === 조부모님 관계 질문 (아버지 쪽) ===
        ('아버지', '아버지'): [
            "{speaker}: \"그럼 {source}의 아버지는 누구야?\"",
            "나: \"{target}야.\""
        ],
        ('아버지', '어머니'): [
            "{speaker}: \"그럼 {source}의 어머니는 누구야?\"",
            "나: \"{target}야.\""
        ],
        ('아버지', '형'): [
            "{speaker}: \"{source}한테 형제 있어? 사진에 있어?\"",
            "나: \"응, 여기 {target}이 형이야.\""
        ],
        ('형', '아내'): [
            "{speaker}: \"{source}는 결혼했어? 사진에 배우자 있어?\"",
            "나: \"응, 여기 {target}이 아내야.\""
        ],
        ('형', '자녀'): [
            "{speaker}: \"{source}에게 자녀 있어? 사진에 있어?\"",
            "나: \"응, 여기 {target}이 아들이야.\""
        ],
        ('아버지', '남동생'): [
            "{speaker}: \"{source}한테 형제 있어? 사진 있어?\"",
            "나: \"응, 여기 {target}이 남동생이야.\""
        ],
        ('남동생', '아내'): [
            "{speaker}: \"{source}는 결혼했어? 사진에 배우자 있어?\"",
            "나: \"응, 여기 {target}이 아내야.\""
        ],
        ('남동생', '자녀'): [
            "{speaker}: \"{source}에게 자녀 있어? 사진에 있어?\"",
            "나: \"응, 여기 {target}이 아들이야.\""
        ],
        ('아버지', '누나'): [
            "{speaker}: \"{source}한테 형제 있어? 사진 있어?\"",
            "나: \"응, 여기 {target}이 누나야.\""
        ],
        ('누나', '남편'): [
            "{speaker}: \"{source}는 결혼했어? 사진에 배우자 있어?\"",
            "나: \"응, 여기 {target}이 남편이야.\""
        ],
        ('누나', '자녀'): [
            "{speaker}: \"{source}에게 자녀 있어? 사진에 있어?\"",
            "나: \"응, 여기 {target}이 아이야.\""
        ],
        ('아버지', '여동생'): [
            "{speaker}: \"{source}한테 형제 있어? 사진 있어?\"",
            "나: \"응, 여기 {target}이 여동생이야.\""
        ],
        ('여동생', '남편'): [
            "{speaker}: \"{source}는 결혼했어? 사진에 배우자 있어?\"",
            "나: \"응, 여기 {target}이 남편이야.\""
        ],
        ('여동생', '자녀'): [
            "{speaker}: \"{source}에게 자녀 있어? 사진에 있어?\"",
            "나: \"응, 여기 {target}이 아이야.\""
        ],

        # === 아버지의 어머니 쪽 (진외가) ===
        ('어머니', '오빠'): [
            "{speaker}: \"{source}한테 남자 형제 있어?\"",
            "나: \"응, 여기 {target}이 오빠야.\""
        ],
        ('오빠', '아내'): [
            "{speaker}: \"{source}는 결혼했어? 배우자는?\"",
            "나: \"응, 여기 {target}이 아내야.\""
        ],
        ('어머니', '남동생'): [
            "{speaker}: \"{source}한테 남자 형제 있어?\"",
            "나: \"응, 여기 {target}이 남동생이야.\""
        ],
        ('어머니', '언니'): [
            "{speaker}: \"{source}한테 여자 형제 있어?\"",
            "나: \"응, 여기 {target}이 언니야.\""
        ],
        ('언니', '남편'): [
            "{speaker}: \"{source}는 결혼했어? 배우자는?\"",
            "나: \"응, 여기 {target}이 남편이야.\""
        ],
        ('언니', '자녀'): [
            "{speaker}: \"{source}에게 자녀 있어? 사진에 있어?\"",
            "나: \"응, 여기 {target}이 조카야.\""
        ],
        ('어머니', '여동생'): [
            "{speaker}: \"{source}한테 여자 형제 있어?\"",
            "나: \"응, 여기 {target}이 여동생이야.\""
        ],

        # === 외조부모님 (어머니 쪽) ===
        ('어머니', '아버지'): [
            "{speaker}: \"그럼 {source}의 아버지는 누구야?\"",
            "나: \"{target}야.\""
        ],
        ('어머니', '어머니'): [
            "{speaker}: \"그럼 {source}의 어머니는 누구야?\"",
            "나: \"{target}야.\""
        ],
        ('오빠', '자녀'): [
            "{speaker}: \"{source}에게 자녀 있어? 사진에 있어?\"",
            "나: \"응, 여기 {target}이 조카야.\""
        ],

        # === 배우자 관계 (시댁: 남편 쪽) ===
        ('나', '남편'): [
            "{speaker}: \"이 사진 속 {target}{particle} 누구야?\"",
            "나: \"아, 그분은 내 남편이야.\""
        ],
        ('남편', '아버지'): [
            "{speaker}: \"그럼 {source}의 아버지는 누구야?\"",
            "나: \"{target}야.\""
        ],
        ('남편', '어머니'): [
            "{speaker}: \"그럼 {source}의 어머니는 누구야?\"",
            "나: \"{target}야.\""
        ],
        ('남편', '형'): [
            "{speaker}: \"{source}한테 남자 형제 있어?\"",
            "나: \"응, 여기 {target}이 형이야.\""
        ],
        ('남편', '남동생'): [
            "{speaker}: \"{source}한테 남자 형제 있어?\"",
            "나: \"응, 여기 {target}이 남동생이야.\""
        ],
        ('남편', '누나'): [
            "{speaker}: \"{source}한테 여자 형제 있어?\"",
            "나: \"응, 여기 {target}이 누나야.\""
        ],
        ('남편', '여동생'): [
            "{speaker}: \"{source}한테 여자 형제 있어?\"",
            "나: \"응, 여기 {target}이 여동생이야.\""
        ],

        # === 배우자 관계 (처가: 아내 쪽) ===
        ('나', '아내'): [
            "{speaker}: \"이 사진 속 {target}{particle} 누구야?\"",
            "나: \"아, 그분은 내 아내야.\""
        ],
        ('아내', '아버지'): [
            "{speaker}: \"그럼 {source}의 아버지는 누구야?\"",
            "나: \"{target}야.\""
        ],
        ('아내', '어머니'): [
            "{speaker}: \"그럼 {source}의 어머니는 누구야?\"",
            "나: \"{target}야.\""
        ],
        ('아내', '오빠'): [
            "{speaker}: \"{source}한테 남자 형제 있어?\"",
            "나: \"응, 여기 {target}이 오빠야.\""
        ],
        ('아내', '남동생'): [
            "{speaker}: \"{source}한테 남자 형제 있어?\"",
            "나: \"응, 여기 {target}이 남동생이야.\""
        ],
        ('아내', '언니'): [
            "{speaker}: \"{source}한테 여자 형제 있어?\"",
            "나: \"응, 여기 {target}이 언니야.\""
        ],
        ('아내', '여동생'): [
            "{speaker}: \"{source}한테 여자 형제 있어?\"",
            "나: \"응, 여기 {target}이 여동생이야.\""
        ],
    }

def get_proper_particle(descriptor):
    if not descriptor:
        return "은"
    last_char = descriptor[-1]
    
    if '가' <= last_char <= '힣':
        has_final = (ord(last_char) - 0xAC00) % 28 != 0
        return "은" if has_final else "는"
    return "은"

def get_all_unique_titles():
    title_map = get_relation_chain_to_title()
    all_titles = set()
    
    for raw_answer in title_map.values():
        if isinstance(raw_answer, list):
            all_titles.update(raw_answer)
        else:
            all_titles.add(raw_answer)
    
    return list(all_titles)

def generate_question(difficulty="Medium"):
    title_map = get_relation_chain_to_title()
    relative_names = get_friend_names()
    descriptors = get_person_descriptors()
    dialogue_map = get_dialogue_templates()

    difficulty_config = {
        "Easy": {
            "num_choices": 5,
            "chain_distribution": {
                3: 0.30,  # 30%
                4: 0.30,  # 30%
                5: 0.40   # 40%
            }
        },
        "Medium": {
            "num_choices": 5,
            "chain_distribution": {
                3: 0.20,  # 20%
                4: 0.30,  # 30%
                5: 0.50   # 50%
            }
        },
        "Hard": {
            "num_choices": 5,
            "chain_distribution": {
                3: 0.10,  # 10%
                4: 0.30,  # 30%
                5: 0.60   # 60%
            }
        }
    }
    
    config = difficulty_config.get(difficulty, difficulty_config["Medium"])
    num_choices = config["num_choices"]
    chain_distribution = config["chain_distribution"]
    
    all_chains = list(title_map.keys())
    chains_by_length = {
        3: [chain for chain in all_chains if len(chain) == 3],
        4: [chain for chain in all_chains if len(chain) == 4],
        5: [chain for chain in all_chains if len(chain) >= 5]  # 5 이상
    }
    
    chain_lengths = list(chain_distribution.keys())
    weights = list(chain_distribution.values())
    selected_length = random.choices(chain_lengths, weights=weights, k=1)[0]
    
    available_chains = chains_by_length.get(selected_length, [])
    
    if not available_chains:
        logging.warning(f"No chains found for length {selected_length}. Using all chains.")
        available_chains = all_chains
    
    relation_chain = random.choice(available_chains)
    raw_answer = title_map[relation_chain]
    
    # print(f"relation_chain: {relation_chain}")
    # print(f"raw_answer: {raw_answer}\n")

    if isinstance(raw_answer, list):
        answer = ", ".join(raw_answer)
        correct_title = random.choice(raw_answer)
    else:
        answer = raw_answer
        correct_title = raw_answer

    person_map = {}
    used_descriptors = set()
    
    for i in range(len(relation_chain) - 1):
        available_descriptors = [desc for desc in descriptors if desc not in used_descriptors]
        
        if not available_descriptors:
            used_descriptors.clear()
            available_descriptors = descriptors
            
        descriptor = random.choice(available_descriptors)
        used_descriptors.add(descriptor)
        person_map[i + 1] = descriptor
    
    dialogue_lines = []
    
    intro_sentences = [
        "오랜만에 친구들이 우리 집에 놀러 왔다. 거실 벽에 걸린 가족사진을 보며 친구들이 이것저것 물어보았다.\n",
        "집들이에 친구들을 초대했다. 새로 꾸민 거실을 구경하던 친구들이 벽에 걸린 가족사진을 발견하고 물었다.\n",
        "동창회에서 만난 친구들과 카페에 앉아 이야기를 나누다가, 휴대폰 속 가족사진을 보여주며 이야기를 꺼냈다.\n"
    ]
    
    dialogue_lines.append(random.choice(intro_sentences))
    
    used_speakers = set()
    
    for i in range(len(relation_chain) - 1):
        source_rel = relation_chain[i]
        target_rel = relation_chain[i+1]
        
        clue_key = (source_rel, target_rel)
        
        if clue_key not in dialogue_map:
            logging.warning(f"Dialogue template not found: {clue_key} (source_rel: {source_rel}, target_rel: {target_rel})")
            return generate_question(difficulty)
        
        available_speakers = [name for name in relative_names if name not in used_speakers]
        if not available_speakers:
            used_speakers.clear()
            available_speakers = relative_names
            
        speaker = random.choice(available_speakers)
        used_speakers.add(speaker)
        
        dialogue_template = dialogue_map[clue_key]
        
        source_index = i
        target_index = i + 1
        
        source_placeholder = "나" if source_index == 0 else person_map[source_index]
        target_placeholder = person_map[target_index]
        
        for line in dialogue_template:
            particle = get_proper_particle(target_placeholder)
            dialogue_line = line.format(
                speaker=speaker, 
                source=source_placeholder, 
                target=target_placeholder,
                particle=particle
            )
            dialogue_lines.append(dialogue_line)

    
    final_person = person_map[len(relation_chain) - 1]
    
    all_titles = get_all_unique_titles()
    
    if isinstance(raw_answer, list):
        wrong_titles = [t for t in all_titles if t not in raw_answer]
    else:
        wrong_titles = [t for t in all_titles if t != raw_answer]
    
    num_wrong_needed = num_choices - 1
    
    if len(wrong_titles) < num_wrong_needed:
        logging.warning(f"Not enough wrong answers for difficulty {difficulty}. "
                       f"Need {num_wrong_needed}, have {len(wrong_titles)}. Regenerating...")
        return generate_question(difficulty)
    
    wrong_choices = random.sample(wrong_titles, num_wrong_needed)
    
    all_options = [correct_title] + wrong_choices
    random.shuffle(all_options)
    
    choices = {}
    correct_letter = None
    for i, title in enumerate(all_options):
        letter = chr(65 + i)
        choices[letter] = title
        if title == correct_title:
            correct_letter = letter
    
    particle = get_proper_particle(final_person)
    dialogue_lines.append(f"\n이때, 나는 {final_person}{particle} 어떻게 불러야 하는가?")
    for i in range(num_choices):
        letter = chr(65 + i)
        dialogue_lines.append(f"{letter}: {choices[letter]}")
    
    question = "\n".join(dialogue_lines)

    explanation = ["[STEP 0] Interpret the given dialogue to understand the relationships between people."]
    temp_chain_str = "me"
    
    for i, rel in enumerate(relation_chain[1:], 1):
        person = person_map[i]
        explanation.append(f"[STEP {i}] From the dialogue, infer that '{person}' is '{temp_chain_str}'s {rel}'.")
        temp_chain_str += f"'s {rel}"
    
    final_step = len(relation_chain)
    explanation.append(f"[STEP {final_step}] Therefore, the final title for the combined relationship '{temp_chain_str}' is '{answer}'.")
    explanation.append(f"[STEP {final_step + 1}] Answer: {correct_letter}")
    

    return question, correct_letter, explanation, choices, difficulty

def create_dataset_files(num_questions_per_difficulty=100):
    import pandas as pd
    import json
    
    difficulties = ["Easy", "Medium", "Hard"]
    
    print(f"Generating kinship problems by difficulty...")
    output = []
    all_generated_data = []
    
    for difficulty in difficulties:
        print(f"\n=== Generating {difficulty} problems ({num_questions_per_difficulty} questions) ===")
        for i in range(num_questions_per_difficulty):
            try:
                q, answer, expl, choices, diff = generate_question(difficulty=difficulty)
                
                output.append({
                    'id': f'kinship_{len(output)}',
                    'question': q,
                    'answer': answer,
                    'solution': "\n".join(expl),
                    'difficulty': diff,
                    'choices': json.dumps(choices, ensure_ascii=False)
                })
                
                all_generated_data.append({
                    'id': f'kinship_{len(all_generated_data)}',
                    'question': q,
                    'answer': answer,
                    'solution': "\n".join(expl),
                    'difficulty': diff,
                    'choices': choices
                })
                
                if (i + 1) % 20 == 0:
                    print(f"  Progress: {i + 1}/{num_questions_per_difficulty}")
                    
            except Exception as e:
                logging.error(f"Error generating {difficulty} question: {e}")
                continue
    
    kinship_df = pd.DataFrame(output)
    
    print(f"\n=== Generation Summary ===")
    print(f"Total problems generated: {len(kinship_df)}")
    print(f"Unique problems: {kinship_df['question'].nunique()}")
    print(f"Duplicate problems: {len(kinship_df) - kinship_df['question'].nunique()}")
    print(f"\nDifficulty breakdown:")
    print(kinship_df['difficulty'].value_counts().sort_index())
    
    # CSV 저장 (question, answer, solution, difficulty만)
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    csv_dir = PROJECT_ROOT / "data" / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = csv_dir / "kinship.csv"
    kinship_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\nCSV file created! -> {csv_path}")
    print(f"CSV columns: {list(kinship_df.columns)}")
    
    # JSONL 저장 (choices 포함)
    json_dir = PROJECT_ROOT / "data" / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    
    jsonl_path = json_dir / "kinship.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item in all_generated_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"JSONL file created! -> {jsonl_path}")
    
    return kinship_df, all_generated_data

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Kinship Puzzle Generator")
    parser.add_argument("--num", type=int, default=100, 
                       help="Number of questions to generate per difficulty level")
    
    args = parser.parse_args()
    
    create_dataset_files(num_questions_per_difficulty=args.num)

    # print("\n=== Sample Problems by Difficulty ===")
    # for diff in ["Easy", "Medium", "Hard"]:
    #     question, answer, explanation, choices, difficulty = generate_question(difficulty=diff)
    #     print(f"\n[{diff}] {len(choices)} choices")
    #     print("[Q]", question)
    #     print("[A]", answer)
    #     print("-" * 60)