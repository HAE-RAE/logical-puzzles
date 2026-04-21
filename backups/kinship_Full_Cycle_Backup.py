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

        # === 시댁 확장 (남편 부모의 부모/형제) ===
        ("나", "남편", "아버지", "아버지"): ["시할아버지", "시조부"],
        ("나", "남편", "아버지", "어머니"): ["시할머니", "시조모"],
        ("나", "남편", "아버지", "형"): ["시백부", "시큰아버지"],
        ("나", "남편", "아버지", "형", "아내"): ["시백모", "시큰어머니"],
        ("나", "남편", "아버지", "남동생"): ["시숙부", "시삼촌"],
        ("나", "남편", "아버지", "남동생", "아내"): "시숙모",
        ("나", "남편", "아버지", "누나"): "시고모",
        ("나", "남편", "아버지", "누나", "남편"): "시고모부",
        ("나", "남편", "아버지", "여동생"): "시고모",
        ("나", "남편", "아버지", "여동생", "남편"): "시고모부",

        # 시댁 외가 (남편 어머니 쪽)
        ("나", "남편", "어머니", "아버지"): ["시외조부", "시외할아버지"],
        ("나", "남편", "어머니", "어머니"): ["시외조모", "시외할머니"],
        ("나", "남편", "어머니", "오빠"): "시외삼촌",
        ("나", "남편", "어머니", "남동생"): "시외삼촌",
        ("나", "남편", "어머니", "언니"): "시이모",
        ("나", "남편", "어머니", "여동생"): "시이모",

        # 시댁 조부모 형제 (6단계)
        ("나", "남편", "아버지", "아버지", "형"): ["시백조부", "시큰할아버지"],
        ("나", "남편", "아버지", "아버지", "형", "아내"): ["시백조모", "시큰할머니"],
        ("나", "남편", "아버지", "아버지", "남동생"): ["시숙조부", "시작은할아버지"],
        ("나", "남편", "아버지", "아버지", "남동생", "아내"): ["시숙조모", "시작은할머니"],
        ("나", "남편", "아버지", "아버지", "누나"): "시대고모",
        ("나", "남편", "아버지", "아버지", "누나", "남편"): "시대고모부",
        ("나", "남편", "아버지", "아버지", "여동생"): "시대고모",
        ("나", "남편", "아버지", "아버지", "여동생", "남편"): "시대고모부",

        # === 처가 확장 (아내 부모의 부모/형제) ===
        ("나", "아내", "아버지", "아버지"): ["처조부", "처할아버지"],
        ("나", "아내", "아버지", "어머니"): ["처조모", "처할머니"],
        ("나", "아내", "아버지", "형"): ["처백부", "처큰아버지"],
        ("나", "아내", "아버지", "형", "아내"): ["처백모", "처큰어머니"],
        ("나", "아내", "아버지", "남동생"): ["처숙부", "처삼촌"],
        ("나", "아내", "아버지", "남동생", "아내"): "처숙모",
        ("나", "아내", "아버지", "누나"): "처고모",
        ("나", "아내", "아버지", "누나", "남편"): "처고모부",
        ("나", "아내", "아버지", "여동생"): "처고모",
        ("나", "아내", "아버지", "여동생", "남편"): "처고모부",

        # 처가 외가 (아내 어머니 쪽)
        ("나", "아내", "어머니", "아버지"): ["처외조부", "처외할아버지"],
        ("나", "아내", "어머니", "어머니"): ["처외조모", "처외할머니"],
        ("나", "아내", "어머니", "오빠"): "처외삼촌",
        ("나", "아내", "어머니", "남동생"): "처외삼촌",
        ("나", "아내", "어머니", "언니"): "처이모",
        ("나", "아내", "어머니", "여동생"): "처이모",

        # 처가 조부모 형제 (6단계)
        ("나", "아내", "아버지", "아버지", "형"): ["처백조부", "처큰할아버지"],
        ("나", "아내", "아버지", "아버지", "형", "아내"): ["처백조모", "처큰할머니"],
        ("나", "아내", "아버지", "아버지", "남동생"): ["처숙조부", "처작은할아버지"],
        ("나", "아내", "아버지", "아버지", "남동생", "아내"): ["처숙조모", "처작은할머니"],
        ("나", "아내", "아버지", "아버지", "누나"): "처대고모",
        ("나", "아내", "아버지", "아버지", "누나", "남편"): "처대고모부",
        ("나", "아내", "아버지", "아버지", "여동생"): "처대고모",
        ("나", "아내", "아버지", "아버지", "여동생", "남편"): "처대고모부",

    }

@functools.lru_cache(None)
def get_friend_names():
    return ["철수", "영희", "준용", "규진", "다솔", "승혁", "영숙", "대곤", "수용"]

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

@functools.lru_cache(None)
def get_reverse_dialogue_templates():
    """
    Reverse dialogue templates where the relationship is stated from the OTHER person's perspective.
    Model must invert the stated relationship to determine the actual kinship.
    e.g. "X는 Y의 남동생이야" → Y is X's 형
    """
    return {
        ('아버지', '아버지'): [
            "{speaker}: \"{target}{particle} {source}이랑 어떤 관계야?\"",
            "나: \"{source}는 {target}의 자녀야.\""
        ],
        ('아버지', '어머니'): [
            "{speaker}: \"{target}{particle} {source}이랑 어떤 관계야?\"",
            "나: \"{source}는 {target}의 자녀야.\""
        ],
        ('아버지', '형'): [
            "{speaker}: \"{source}한테 형제 있어?\"",
            "나: \"응, {source}는 {target}의 남동생이야.\""
        ],
        ('아버지', '남동생'): [
            "{speaker}: \"{source}한테 형제 있어?\"",
            "나: \"응, {source}는 {target}의 형이야.\""
        ],
        ('아버지', '누나'): [
            "{speaker}: \"{source}한테 형제 있어?\"",
            "나: \"응, {source}는 {target}의 남동생이야.\""
        ],
        ('아버지', '여동생'): [
            "{speaker}: \"{source}한테 형제 있어?\"",
            "나: \"응, {source}는 {target}의 오빠야.\""
        ],
        ('형', '아내'): [
            "{speaker}: \"{source}는 결혼했어?\"",
            "나: \"응, {target}의 남편이 {source}야.\""
        ],
        ('남동생', '아내'): [
            "{speaker}: \"{source}는 결혼했어?\"",
            "나: \"응, {target}의 남편이 {source}야.\""
        ],
        ('누나', '남편'): [
            "{speaker}: \"{source}는 결혼했어?\"",
            "나: \"응, {target}의 아내가 {source}야.\""
        ],
        ('여동생', '남편'): [
            "{speaker}: \"{source}는 결혼했어?\"",
            "나: \"응, {target}의 아내가 {source}야.\""
        ],
        ('어머니', '아버지'): [
            "{speaker}: \"{target}{particle} {source}이랑 어떤 관계야?\"",
            "나: \"{source}는 {target}의 자녀야.\""
        ],
        ('어머니', '어머니'): [
            "{speaker}: \"{target}{particle} {source}이랑 어떤 관계야?\"",
            "나: \"{source}는 {target}의 자녀야.\""
        ],
        ('어머니', '오빠'): [
            "{speaker}: \"{source}한테 남자 형제 있어?\"",
            "나: \"응, {source}는 {target}의 여동생이야.\""
        ],
        ('어머니', '남동생'): [
            "{speaker}: \"{source}한테 남자 형제 있어?\"",
            "나: \"응, {source}는 {target}의 누나야.\""
        ],
        ('어머니', '언니'): [
            "{speaker}: \"{source}한테 여자 형제 있어?\"",
            "나: \"응, {source}는 {target}의 여동생이야.\""
        ],
        ('어머니', '여동생'): [
            "{speaker}: \"{source}한테 여자 형제 있어?\"",
            "나: \"응, {source}는 {target}의 언니야.\""
        ],
        ('오빠', '아내'): [
            "{speaker}: \"{source}는 결혼했어?\"",
            "나: \"응, {target}의 남편이 {source}야.\""
        ],
        ('언니', '남편'): [
            "{speaker}: \"{source}는 결혼했어?\"",
            "나: \"응, {target}의 아내가 {source}야.\""
        ],
        ('남편', '아버지'): [
            "{speaker}: \"{target}{particle} {source}이랑 어떤 관계야?\"",
            "나: \"{source}는 {target}의 아들이야.\""
        ],
        ('남편', '어머니'): [
            "{speaker}: \"{target}{particle} {source}이랑 어떤 관계야?\"",
            "나: \"{source}는 {target}의 아들이야.\""
        ],
        ('남편', '형'): [
            "{speaker}: \"{source}한테 남자 형제 있어?\"",
            "나: \"응, {source}는 {target}의 남동생이야.\""
        ],
        ('남편', '남동생'): [
            "{speaker}: \"{source}한테 남자 형제 있어?\"",
            "나: \"응, {source}는 {target}의 형이야.\""
        ],
        ('남편', '누나'): [
            "{speaker}: \"{source}한테 여자 형제 있어?\"",
            "나: \"응, {source}는 {target}의 남동생이야.\""
        ],
        ('남편', '여동생'): [
            "{speaker}: \"{source}한테 여자 형제 있어?\"",
            "나: \"응, {source}는 {target}의 오빠야.\""
        ],
        ('아내', '아버지'): [
            "{speaker}: \"{target}{particle} {source}이랑 어떤 관계야?\"",
            "나: \"{source}는 {target}의 딸이야.\""
        ],
        ('아내', '어머니'): [
            "{speaker}: \"{target}{particle} {source}이랑 어떤 관계야?\"",
            "나: \"{source}는 {target}의 딸이야.\""
        ],
        ('아내', '오빠'): [
            "{speaker}: \"{source}한테 남자 형제 있어?\"",
            "나: \"응, {source}는 {target}의 여동생이야.\""
        ],
        ('아내', '남동생'): [
            "{speaker}: \"{source}한테 남자 형제 있어?\"",
            "나: \"응, {source}는 {target}의 누나야.\""
        ],
        ('아내', '언니'): [
            "{speaker}: \"{source}한테 여자 형제 있어?\"",
            "나: \"응, {source}는 {target}의 여동생이야.\""
        ],
        ('아내', '여동생'): [
            "{speaker}: \"{source}한테 여자 형제 있어?\"",
            "나: \"응, {source}는 {target}의 언니야.\""
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

@functools.lru_cache(None)
def get_title_to_family_branch():
    """
    Maps each kinship title to its family branch for similar distractor selection.
    Branches: 친가 (father's side), 외가 (mother's side), 시댁 (husband's family), 처가 (wife's family)
    """
    return {
        # 친가 (아버지 쪽)
        "친할아버지": "친가", "친할머니": "친가", "큰아버지": "친가", "백부": "친가",
        "큰어머니": "친가", "백모": "친가", "종형제": "친가", "사촌": "친가",
        "작은아버지": "친가", "숙부": "친가", "삼촌": "친가", "작은어머니": "친가",
        "숙모": "친가", "고모": "친가", "고모부": "친가", "내종형제": "친가",
        "고종사촌": "친가",
        "증조할아버지": "친가", "증조할머니": "친가", "큰할아버지": "친가",
        "백조부": "친가", "큰할머니": "친가", "백조모": "친가", "작은할아버지": "친가",
        "숙조부": "친가", "작은할머니": "친가", "숙조모": "친가",
        "고모할머니": "친가", "대고모": "친가", "고모할아버지": "친가", "대고모부": "친가",
        # 진외가 (아버지 어머니 쪽) - 친가와 유사
        "진외종조부": "친가", "진외할아버지": "친가", "진외이모할머니": "친가",
        "대이모": "친가", "진외이모할아버지": "친가", "대이모부": "친가",
        # 외가 (어머니 쪽)
        "외할아버지": "외가", "외조부": "외가", "외할머니": "외가", "외조모": "외가",
        "큰외삼촌": "외가", "외숙부": "외가", "외삼촌": "외가", "작은외삼촌": "외가",
        "큰외숙모": "외가", "외숙모": "외가", "작은외숙모": "외가", "외사촌": "외가", "외종형제": "외가", "큰이모": "외가",
        "이모": "외가", "큰이모부": "외가", "이모부": "외가", "이종사촌": "외가",
        "이종형제": "외가", "작은이모": "외가", "작은이모부": "외가",
        "증조외할아버지": "외가", "외증조부": "외가", "증조외할머니": "외가",
        "외증조모": "외가", "큰외할아버지": "외가", "외종조부": "외가",
        "큰외할머니": "외가", "외종조모": "외가", "작은외할아버지": "외가",
        "작은외할머니": "외가", "고모외할머니": "외가", "외대고모": "외가",
        "고모외할아버지": "외가", "외대고모부": "외가",
        "외삼촌할아버지": "외가", "큰이모할머니": "외가", "이모할머니": "외가",
        "작은이모할머니": "외가", "이모할아버지": "외가",
        # 시댁 (남편 쪽)
        "시아버지": "시댁", "아버님": "시댁", "시어머니": "시댁", "어머님": "시댁",
        "아주버님": "시댁", "시숙": "시댁", "형님": "시댁", "도련님": "시댁",
        "서방님": "시댁", "동서": "시댁", "시누이": "시댁", "아가씨": "시댁",
        "고모부님": "시댁",
        # 시댁 확장
        "시할아버지": "시댁", "시조부": "시댁", "시할머니": "시댁", "시조모": "시댁",
        "시백부": "시댁", "시큰아버지": "시댁", "시백모": "시댁", "시큰어머니": "시댁",
        "시숙부": "시댁", "시삼촌": "시댁", "시숙모": "시댁",
        "시고모": "시댁", "시고모부": "시댁",
        "시외조부": "시댁", "시외할아버지": "시댁", "시외조모": "시댁", "시외할머니": "시댁",
        "시외삼촌": "시댁", "시이모": "시댁",
        "시백조부": "시댁", "시큰할아버지": "시댁", "시백조모": "시댁", "시큰할머니": "시댁",
        "시숙조부": "시댁", "시작은할아버지": "시댁", "시숙조모": "시댁", "시작은할머니": "시댁",
        "시대고모": "시댁", "시대고모부": "시댁",
        # 처가 (아내 쪽)
        "장인": "처가", "장인어른": "처가", "장모": "처가", "장모님": "처가",
        "아주머니": "처가", "처남댁": "처가", "처남": "처가", "처형": "처가",
        "처제": "처가", "제부": "처가", "서방": "처가",
        # 처가 확장
        "처조부": "처가", "처할아버지": "처가", "처조모": "처가", "처할머니": "처가",
        "처백부": "처가", "처큰아버지": "처가", "처백모": "처가", "처큰어머니": "처가",
        "처숙부": "처가", "처삼촌": "처가", "처숙모": "처가",
        "처고모": "처가", "처고모부": "처가",
        "처외조부": "처가", "처외할아버지": "처가", "처외조모": "처가", "처외할머니": "처가",
        "처외삼촌": "처가", "처이모": "처가",
        "처백조부": "처가", "처큰할아버지": "처가", "처백조모": "처가", "처큰할머니": "처가",
        "처숙조부": "처가", "처작은할아버지": "처가", "처숙조모": "처가", "처작은할머니": "처가",
        "처대고모": "처가", "처대고모부": "처가",
    }

def get_similar_titles(correct_title, exclude_titles, n, branch_map=None):
    """
    Returns n titles from the same family branch as correct_title (confusing distractors).
    exclude_titles: set of titles to exclude (correct answer variants)
    """
    if branch_map is None:
        branch_map = get_title_to_family_branch()
    correct_branch = branch_map.get(correct_title, "친가")  # default to 친가 if unknown
    all_titles = get_all_unique_titles()
    similar = [t for t in all_titles if t not in exclude_titles and branch_map.get(t, "") == correct_branch]
    if len(similar) >= n:
        return random.sample(similar, n)
    return similar

def get_confusable_titles(correct_title, exclude_titles, n):
    """
    Returns near-miss distractors that are lexically/semantically close to correct_title.
    Heuristic scoring prioritizes same branch + surface-form similarity.
    """
    if n <= 0:
        return []

    branch_map = get_title_to_family_branch()
    all_titles = get_all_unique_titles()
    correct_branch = branch_map.get(correct_title, "")

    def normalize_title(title):
        for token in ("큰", "작은", "대", "친", "진", "외", "시", "처"):
            title = title.replace(token, "")
        return title

    normalized_correct = normalize_title(correct_title)
    suffix_correct = correct_title[-1] if correct_title else ""
    chars_correct = set(correct_title)

    candidates = []
    for title in all_titles:
        if title in exclude_titles:
            continue

        score = 0
        if branch_map.get(title, "") == correct_branch:
            score += 4
        if title and title[-1] == suffix_correct:
            score += 3
        if normalize_title(title) == normalized_correct:
            score += 4
        score += len(chars_correct.intersection(set(title)))

        candidates.append((score, title))

    if not candidates:
        return []

    candidates.sort(key=lambda x: x[0], reverse=True)
    top_k = candidates[: max(n * 3, n)]
    pool = [title for _, title in top_k]
    if len(pool) <= n:
        return pool
    return random.sample(pool, n)

def get_prefix_near_miss_titles(target_chain, title_map, exclude_titles, n):
    """
    Returns distractor titles from chains that share the same prefix and only
    diverge near the end (strong near-miss candidates).
    """
    if n <= 0 or not target_chain or len(target_chain) < 3:
        return []

    target_chain = tuple(target_chain)
    branch_map = get_title_to_family_branch()
    all_candidates = []

    def append_titles(raw_titles, chain_score):
        titles = raw_titles if isinstance(raw_titles, list) else [raw_titles]
        for t in titles:
            if t in exclude_titles:
                continue
            all_candidates.append((chain_score, t))

    # 1) Strongest: same length + same prefix except last relation
    strong_prefix = target_chain[:-1]
    for chain, raw in title_map.items():
        if chain == target_chain:
            continue
        if len(chain) != len(target_chain):
            continue
        if chain[:-1] != strong_prefix:
            continue
        append_titles(raw, chain_score=3)

    # 2) Fallback: same length + same prefix except last two relations
    if len(target_chain) >= 4:
        mid_prefix = target_chain[:-2]
        for chain, raw in title_map.items():
            if chain == target_chain:
                continue
            if len(chain) != len(target_chain):
                continue
            if chain[:-2] != mid_prefix:
                continue
            append_titles(raw, chain_score=2)

    # 3) Fallback: one step shorter/longer chains with same leading prefix
    base_prefix = target_chain[:-1]
    for chain, raw in title_map.items():
        if chain == target_chain:
            continue
        if len(chain) not in (len(target_chain) - 1, len(target_chain) + 1):
            continue
        compare_len = min(len(base_prefix), len(chain) - 1)
        if compare_len <= 1:
            continue
        if chain[:compare_len] != base_prefix[:compare_len]:
            continue
        append_titles(raw, chain_score=1)

    if not all_candidates:
        return []

    # Prefer same family branch for stronger confusion.
    target_titles = title_map.get(target_chain)
    if isinstance(target_titles, list):
        anchor_title = target_titles[0] if target_titles else ""
    else:
        anchor_title = target_titles
    anchor_branch = branch_map.get(anchor_title, "")

    scored = []
    seen = set()
    for chain_score, title in all_candidates:
        if title in seen:
            continue
        seen.add(title)
        branch_bonus = 1 if anchor_branch and branch_map.get(title, "") == anchor_branch else 0
        scored.append((chain_score + branch_bonus, title))

    scored.sort(key=lambda x: x[0], reverse=True)
    pool = [t for _, t in scored]
    if len(pool) <= n:
        return pool
    return random.sample(pool[: max(n * 3, n)], n)

def generate_noise_dialogues(relation_chain, person_map, used_descriptors,
                             dialogue_map, relative_names, used_speakers,
                             descriptors, num_noise):
    """Generate noise (red herring) dialogue blocks unrelated to the answer chain."""
    if num_noise <= 0:
        return []

    chain_steps = set()
    for i in range(len(relation_chain) - 1):
        chain_steps.add((relation_chain[i], relation_chain[i + 1]))

    possible_branches = []
    for idx in range(1, len(relation_chain) - 1):
        source_rel = relation_chain[idx]
        for key in dialogue_map.keys():
            if key[0] == source_rel and key not in chain_steps:
                possible_branches.append((idx, key))

    noise_blocks = []
    for _ in range(num_noise):
        if not possible_branches:
            break

        branch_idx, clue_key = random.choice(possible_branches)
        source_placeholder = person_map[branch_idx]

        available_descs = [d for d in descriptors if d not in used_descriptors]
        if not available_descs:
            used_descriptors.clear()
            available_descs = list(descriptors)
        noise_descriptor = random.choice(available_descs)
        used_descriptors.add(noise_descriptor)

        available_spk = [n for n in relative_names if n not in used_speakers]
        if not available_spk:
            used_speakers.clear()
            available_spk = list(relative_names)
        speaker = random.choice(available_spk)
        used_speakers.add(speaker)

        template = dialogue_map[clue_key]
        lines = []
        for line in template:
            particle = get_proper_particle(noise_descriptor)
            dialogue_line = line.format(
                speaker=speaker,
                source=source_placeholder,
                target=noise_descriptor,
                particle=particle
            )
            lines.append(dialogue_line)
        noise_blocks.append(lines)

    return noise_blocks

def generate_question(difficulty="Medium"):
    title_map = get_relation_chain_to_title()
    relative_names = get_friend_names()
    descriptors = get_person_descriptors()
    dialogue_map = get_dialogue_templates()

    difficulty_lower = difficulty.lower() if isinstance(difficulty, str) else difficulty
    difficulty_config = {
        "easy": {
            "num_choices": 6,
            "chain_distribution": {
                3: 0.60,
                4: 0.40,
            },
            "num_noise_dialogues": 0,
            "near_miss_distractor_ratio": 0.0,
            "similar_distractor_ratio": 0.20,
            "confusable_distractor_ratio": 0.0,
            "shuffle_mode": "none",
            "ask_intermediate_prob": 0.0,
            "reverse_prob": 0.0,
            "force_reverse_min_steps": 0,
            "correction_prob": 0.0,
        },
        "medium": {
            "num_choices": 12,
            "chain_distribution": {
                3: 0.10,
                4: 0.40,
                5: 0.40,
                6: 0.10,
            },
            "num_noise_dialogues": 2,
            "near_miss_distractor_ratio": 0.20,
            "similar_distractor_ratio": 0.60,
            "confusable_distractor_ratio": 0.30,
            "shuffle_mode": "partial",
            "ask_intermediate_prob": 0.0,
            "reverse_prob": 0.40,
            "force_reverse_min_steps": 1,
            "correction_prob": 0.0,
        },
        "hard": {
            "num_choices": 16,
            "chain_distribution": {
                4: 0.30,
                5: 0.50,
                6: 0.20,
            },
            "num_noise_dialogues": 5,
            "near_miss_distractor_ratio": 0.55,
            "similar_distractor_ratio": 1.0,
            "confusable_distractor_ratio": 1.0,
            "shuffle_mode": "full",
            "ask_intermediate_prob": 0.40,
            "reverse_prob": 0.85,
            "force_reverse_min_steps": 3,
            "correction_prob": 0.0,
        }
    }
    
    config = difficulty_config.get(difficulty_lower, difficulty_config["medium"])
    num_choices = config["num_choices"]
    chain_distribution = config["chain_distribution"]
    
    all_chains = list(title_map.keys())
    chains_by_length = {
        3: [chain for chain in all_chains if len(chain) == 3],
        4: [chain for chain in all_chains if len(chain) == 4],
        5: [chain for chain in all_chains if len(chain) == 5],
        6: [chain for chain in all_chains if len(chain) >= 6],
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
    
    if isinstance(raw_answer, list):
        answer = ", ".join(raw_answer)
        correct_title = random.choice(raw_answer)
        exclude_set = set(raw_answer)
    else:
        answer = raw_answer
        correct_title = raw_answer
        exclude_set = {raw_answer}

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
    
    intro_sentences = [
        "오랜만에 친구들이 우리 집에 놀러 왔다. 거실 벽에 걸린 가족사진을 보며 친구들이 이것저것 물어보았다.\n",
        "집들이에 친구들을 초대했다. 새로 꾸민 거실을 구경하던 친구들이 벽에 걸린 가족사진을 발견하고 물었다.\n",
        "동창회에서 만난 친구들과 카페에 앉아 이야기를 나누다가, 휴대폰 속 가족사진을 보여주며 이야기를 꺼냈다.\n"
    ]
    
    used_speakers = set()
    real_dialogue_blocks = []
    reverse_map = get_reverse_dialogue_templates()
    reverse_prob = config.get("reverse_prob", 0.0)
    force_reverse_min_steps = config.get("force_reverse_min_steps", 0)
    reversible_indices = [
        i for i in range(len(relation_chain) - 1)
        if (relation_chain[i], relation_chain[i + 1]) in reverse_map
    ]
    forced_reverse_indices = set()
    if reversible_indices and force_reverse_min_steps > 0:
        forced_reverse_indices = set(
            random.sample(
                reversible_indices,
                min(force_reverse_min_steps, len(reversible_indices)),
            )
        )
    
    
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
        
        use_reverse = (
            clue_key in reverse_map and (
                i in forced_reverse_indices
                or (reverse_prob > 0 and random.random() < reverse_prob)
            )
        )
        dialogue_template = reverse_map[clue_key] if use_reverse else dialogue_map[clue_key]
        
        source_index = i
        target_index = i + 1
        
        source_placeholder = "나" if source_index == 0 else person_map[source_index]
        target_placeholder = person_map[target_index]
        particle = get_proper_particle(target_placeholder)
        
        block_lines = []
        for line in dialogue_template:
            dialogue_line = line.format(
                speaker=speaker,
                source=source_placeholder,
                target=target_placeholder,
                particle=particle
            )
            block_lines.append(dialogue_line)
                
        real_dialogue_blocks.append(block_lines)

    num_noise = config.get("num_noise_dialogues", 0)
    noise_blocks = generate_noise_dialogues(
        relation_chain, person_map, used_descriptors,
        dialogue_map, relative_names, used_speakers,
        descriptors, num_noise
    )

    all_blocks = list(real_dialogue_blocks)
    for noise_block in noise_blocks:
        if len(all_blocks) > 1:
            insert_pos = random.randint(1, len(all_blocks) - 1)
        else:
            insert_pos = 1
        all_blocks.insert(insert_pos, noise_block)

    shuffle_mode = config.get("shuffle_mode", "none")
    if shuffle_mode == "partial" and len(all_blocks) > 2:
        first_block = all_blocks[0]
        rest = all_blocks[1:]
        random.shuffle(rest)
        all_blocks = [first_block] + rest
    elif shuffle_mode == "full" and len(all_blocks) > 1:
        random.shuffle(all_blocks)

    dialogue_lines = [random.choice(intro_sentences)]
    for block in all_blocks:
        dialogue_lines.extend(block)

    ask_intermediate_prob = config.get("ask_intermediate_prob", 0.0)
    ask_person_index = len(relation_chain) - 1

    if ask_intermediate_prob > 0 and random.random() < ask_intermediate_prob and len(relation_chain) > 3:
        intermediate_idx = len(relation_chain) - 2
        partial_chain = relation_chain[:intermediate_idx + 1]
        if partial_chain in title_map:
            ask_person_index = intermediate_idx
            raw_answer = title_map[partial_chain]
            if isinstance(raw_answer, list):
                answer = ", ".join(raw_answer)
                correct_title = random.choice(raw_answer)
                exclude_set = set(raw_answer)
            else:
                answer = raw_answer
                correct_title = raw_answer
                exclude_set = {raw_answer}

    target_person = person_map[ask_person_index]
    target_chain = relation_chain[:ask_person_index + 1]
    
    all_titles = get_all_unique_titles()
    
    wrong_titles = [t for t in all_titles if t not in exclude_set]
    num_wrong_needed = num_choices - 1

    near_miss_ratio = config.get("near_miss_distractor_ratio", 0.0)
    confusable_ratio = config.get("confusable_distractor_ratio", 0.0)
    similar_ratio = config.get("similar_distractor_ratio", 0.5)

    budget = num_wrong_needed
    n_near_miss_needed = min(round(num_wrong_needed * near_miss_ratio), budget)
    budget -= n_near_miss_needed
    n_confusable_needed = min(round(num_wrong_needed * confusable_ratio), budget)
    budget -= n_confusable_needed
    n_similar_needed = min(round(num_wrong_needed * similar_ratio), budget)
    budget -= n_similar_needed
    n_random_needed = budget

    near_miss = get_prefix_near_miss_titles(target_chain, title_map, exclude_set, n_near_miss_needed)
    wrong_from_near_miss = near_miss[:n_near_miss_needed]
    if len(wrong_from_near_miss) < n_near_miss_needed:
        n_random_needed += n_near_miss_needed - len(wrong_from_near_miss)

    exclude_plus_near = set(exclude_set).union(wrong_from_near_miss)
    confusable = get_confusable_titles(correct_title, exclude_plus_near, n_confusable_needed)
    wrong_from_confusable = confusable[:n_confusable_needed]
    if len(wrong_from_confusable) < n_confusable_needed:
        n_random_needed += n_confusable_needed - len(wrong_from_confusable)

    exclude_plus = set(exclude_set).union(set(wrong_from_near_miss)).union(set(wrong_from_confusable))
    similar = get_similar_titles(correct_title, exclude_plus, n_similar_needed)
    wrong_from_similar = similar[:n_similar_needed]
    if len(wrong_from_similar) < n_similar_needed:
        n_random_needed += n_similar_needed - len(wrong_from_similar)

    random_pool = [
        t for t in wrong_titles
        if t not in set(wrong_from_near_miss).union(set(wrong_from_confusable)).union(set(wrong_from_similar))
    ]
    wrong_from_random = random.sample(random_pool, min(n_random_needed, len(random_pool)))
    wrong_choices = wrong_from_near_miss + wrong_from_confusable + wrong_from_similar + wrong_from_random
    if len(wrong_choices) > num_wrong_needed:
        wrong_choices = wrong_choices[:num_wrong_needed]

    if len(wrong_choices) < num_wrong_needed:
        remaining = [t for t in wrong_titles if t not in set(wrong_choices)]
        wrong_choices += random.sample(remaining, min(num_wrong_needed - len(wrong_choices), len(remaining)))

    if len(wrong_choices) < num_wrong_needed:
        logging.warning(f"Not enough wrong answers for difficulty {difficulty}. "
                       f"Need {num_wrong_needed}, have {len(wrong_choices)}. Regenerating...")
        return generate_question(difficulty)
    
    all_options = [correct_title] + wrong_choices
    random.shuffle(all_options)
    
    choices = {}
    correct_letter = None
    for i, title in enumerate(all_options):
        letter = chr(65 + i)
        choices[letter] = title
        if title == correct_title:
            correct_letter = letter
    
    particle = get_proper_particle(target_person)
    dialogue_lines.append(f"\n이때, 나는 {target_person}{particle} 어떻게 불러야 하는가?")
    for i in range(num_choices):
        letter = chr(65 + i)
        dialogue_lines.append(f"{letter}: {choices[letter]}")
    
    question = "\n".join(dialogue_lines)

    # ✅ 해설(Explanation)에 함정 언급 추가
    explanation = ["[STEP 0] Interpret the given dialogue to understand the relationships between people."]
    temp_chain_str = "me"
    
    for i, rel in enumerate(target_chain[1:], 1):
        person = person_map[i]
        
        explanation.append(f"[STEP {i}] From the dialogue, infer that '{person}' is '{temp_chain_str}'s {rel}'.")
            
        temp_chain_str += f"'s {rel}"
    
    final_step = len(target_chain)
    explanation.append(f"[STEP {final_step}] Therefore, the final title for the combined relationship '{temp_chain_str}' is '{answer}'.")
    explanation.append(f"[STEP {final_step + 1}] Answer: {correct_letter}")
    

    return question, correct_letter, explanation, choices, difficulty.lower()

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
    json_dir = PROJECT_ROOT / "data" / "jsonl"
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