import random
from pathlib import Path
import functools
import logging
import json
import pandas as pd
from enum import Enum
from kinship import get_friend_names, get_dialogue_templates, get_proper_particle

logging.basicConfig(level=logging.WARNING, format='[%(levelname)s] %(message)s')


class Gender(Enum):
    MALE = "male"
    FEMALE = "female"

class AgeGroup(Enum):
    SENIOR = "senior"
    ADULT = "adult"
    YOUNG_ADULT = "young_adult"
    CHILD = "child"

@functools.lru_cache(None)
def get_relation_chain_to_title():
    return {
        # === 기본 관계 ===
        ("나", "아버지"): {
            "titles": "아버지", "gender": Gender.MALE, "age": AgeGroup.ADULT
        },
        ("나", "어머니"): {
            "titles": "어머니", "gender": Gender.FEMALE, "age": AgeGroup.ADULT
        },
        ("나", "남편"): {
            "titles": "남편", "gender": Gender.MALE, "age": AgeGroup.YOUNG_ADULT
        },
        ("나", "아내"): {
            "titles": "아내", "gender": Gender.FEMALE, "age": AgeGroup.YOUNG_ADULT
        },
        
        # === 형제자매 관계 ===
        ("나", "형"): {
            "titles": "형", "gender": Gender.MALE, "age": AgeGroup.YOUNG_ADULT
        },
        ("나", "누나"): {
            "titles": "누나", "gender": Gender.FEMALE, "age": AgeGroup.YOUNG_ADULT
        },
        ("나", "남동생"): {
            "titles": "남동생", "gender": Gender.MALE, "age": AgeGroup.YOUNG_ADULT
        },
        ("나", "여동생"): {
            "titles": "여동생", "gender": Gender.FEMALE, "age": AgeGroup.YOUNG_ADULT
        },
        
        # === 친가 (아버지 쪽) ===
        ("나", "아버지", "아버지"): {
            "titles": "친할아버지", "gender": Gender.MALE, "age": AgeGroup.SENIOR
        },
        ("나", "아버지", "어머니"): {
            "titles": "친할머니", "gender": Gender.FEMALE, "age": AgeGroup.SENIOR
        },

        ("나", "아버지", "형"): {
            "titles": ["큰아버지", "백부"], "gender": Gender.MALE, "age": AgeGroup.ADULT
        },
        ("나", "아버지", "형", "아내"): {
            "titles": ["큰어머니", "백모"], "gender": Gender.FEMALE, "age": AgeGroup.ADULT
        },
        ("나", "아버지", "형", "자녀"): {
            "titles": ["종형제", "사촌"], "gender": Gender.MALE, "age": AgeGroup.YOUNG_ADULT
        },
        ("나", "아버지", "남동생"): {
            "titles": ["작은아버지", "숙부", "삼촌"], "gender": Gender.MALE, "age": AgeGroup.ADULT
        },
        ("나", "아버지", "남동생", "아내"): {
            "titles": ["작은어머니", "숙모"], "gender": Gender.FEMALE, "age": AgeGroup.ADULT
        },
        ("나", "아버지", "남동생", "자녀"): {
            "titles": ["종형제", "사촌"], "gender": Gender.MALE, "age": AgeGroup.YOUNG_ADULT
        },
        ("나", "아버지", "누나"): {
            "titles": "고모", "gender": Gender.FEMALE, "age": AgeGroup.ADULT
        },
        ("나", "아버지", "누나", "남편"): {
            "titles": "고모부", "gender": Gender.MALE, "age": AgeGroup.ADULT
        },
        ("나", "아버지", "누나", "자녀"): {
            "titles": ["내종형제", "고종사촌"], "gender": Gender.MALE, "age": AgeGroup.YOUNG_ADULT
        },
        ("나", "아버지", "여동생"): {
            "titles": "고모", "gender": Gender.FEMALE, "age": AgeGroup.ADULT
        },
        ("나", "아버지", "여동생", "남편"): {
            "titles": "고모부", "gender": Gender.MALE, "age": AgeGroup.ADULT
        },
        ("나", "아버지", "여동생", "자녀"): {
            "titles": ["내종형제", "고종사촌"], "gender": Gender.MALE, "age": AgeGroup.YOUNG_ADULT
        },

        # === 증조/방계 (증조부모님) ===
        ("나", "아버지", "아버지", "아버지"): {
            "titles": "증조할아버지", "gender": Gender.MALE, "age": AgeGroup.SENIOR
        },
        ("나", "아버지", "아버지", "어머니"): {
            "titles": "증조할머니", "gender": Gender.FEMALE, "age": AgeGroup.SENIOR
        },

        ("나", "아버지", "아버지", "형"): {
            "titles": ["큰할아버지", "백조부"], "gender": Gender.MALE, "age": AgeGroup.SENIOR
        },
        ("나", "아버지", "아버지", "형", "아내"): {
            "titles": ["큰할머니", "백조모"], "gender": Gender.FEMALE, "age": AgeGroup.SENIOR
        },
        ("나", "아버지", "아버지", "남동생"): {
            "titles": ["작은할아버지", "숙조부"], "gender": Gender.MALE, "age": AgeGroup.SENIOR
        },
        ("나", "아버지", "아버지", "남동생", "아내"): {
            "titles": ["작은할머니", "숙조모"], "gender": Gender.FEMALE, "age": AgeGroup.SENIOR
        },
        ("나", "아버지", "아버지", "누나"): {
            "titles": ["고모할머니", "대고모"], "gender": Gender.FEMALE, "age": AgeGroup.SENIOR
        },
        ("나", "아버지", "아버지", "누나", "남편"): {
            "titles": ["고모할아버지", "대고모부"], "gender": Gender.MALE, "age": AgeGroup.SENIOR
        },
        ("나", "아버지", "아버지", "여동생"): {
            "titles": ["고모할머니", "대고모"], "gender": Gender.FEMALE, "age": AgeGroup.SENIOR
        },
        ("나", "아버지", "아버지", "여동생", "남편"): {
            "titles": ["고모할아버지", "대고모부"], "gender": Gender.MALE, "age": AgeGroup.SENIOR
        },

        # === 진외가 (아버지의 외가) ===
        ("나", "아버지", "어머니", "오빠"): {
            "titles": ["진외종조부", "진외할아버지"], "gender": Gender.MALE, "age": AgeGroup.SENIOR
        },
        ("나", "아버지", "어머니", "남동생"): {
            "titles": ["진외종조부", "진외할아버지"], "gender": Gender.MALE, "age": AgeGroup.SENIOR
        },
        ("나", "아버지", "어머니", "언니"): {
            "titles": ["진외이모할머니", "대이모"], "gender": Gender.FEMALE, "age": AgeGroup.SENIOR
        },
        ("나", "아버지", "어머니", "언니", "남편"): {
            "titles": ["진외이모할아버지", "대이모부"], "gender": Gender.MALE, "age": AgeGroup.SENIOR
        },
        ("나", "아버지", "어머니", "여동생"): {
            "titles": ["진외이모할머니", "대이모"], "gender": Gender.FEMALE, "age": AgeGroup.SENIOR
        },
        ("나", "아버지", "어머니", "여동생", "남편"): {
            "titles": ["진외이모할아버지", "대이모부"], "gender": Gender.MALE, "age": AgeGroup.SENIOR
        },

        # === 외가 (어머니 쪽) ===
        ("나", "어머니", "아버지"): {
            "titles": ["외할아버지", "외조부"], "gender": Gender.MALE, "age": AgeGroup.SENIOR
        },
        ("나", "어머니", "어머니"): {
            "titles": ["외할머니", "외조모"], "gender": Gender.FEMALE, "age": AgeGroup.SENIOR
        },

        ("나", "어머니", "오빠"): {
            "titles": ["큰외삼촌", "외숙부", "외삼촌"], "gender": Gender.MALE, "age": AgeGroup.ADULT
        },
        ("나", "어머니", "오빠", "아내"): {
            "titles": ["큰외숙모", "외숙모"], "gender": Gender.FEMALE, "age": AgeGroup.ADULT
        },
        ("나", "어머니", "오빠", "자녀"): {
            "titles": ["외사촌", "외종형제"], "gender": Gender.MALE, "age": AgeGroup.YOUNG_ADULT
        },
        ("나", "어머니", "남동생"): {
            "titles": ["작은외삼촌", "외숙부", "외삼촌"], "gender": Gender.MALE, "age": AgeGroup.ADULT
        },
        ("나", "어머니", "남동생", "아내"): {
            "titles": ["작은외숙모", "외숙모"], "gender": Gender.FEMALE, "age": AgeGroup.ADULT
        },
        ("나", "어머니", "남동생", "자녀"): {
            "titles": ["외사촌", "외종형제"], "gender": Gender.MALE, "age": AgeGroup.YOUNG_ADULT
        },
        ("나", "어머니", "언니"): {
            "titles": ["큰이모", "이모"], "gender": Gender.FEMALE, "age": AgeGroup.ADULT
        },
        ("나", "어머니", "언니", "남편"): {
            "titles": ["큰이모부", "이모부"], "gender": Gender.MALE, "age": AgeGroup.ADULT
        },
        ("나", "어머니", "언니", "자녀"): {
            "titles": ["이종사촌", "이종형제"], "gender": Gender.MALE, "age": AgeGroup.YOUNG_ADULT
        },
        ("나", "어머니", "여동생"): {
            "titles": ["작은이모", "이모"], "gender": Gender.FEMALE, "age": AgeGroup.ADULT
        },
        ("나", "어머니", "여동생", "남편"): {
            "titles": ["작은이모부", "이모부"], "gender": Gender.MALE, "age": AgeGroup.ADULT
        },
        ("나", "어머니", "여동생", "자녀"): {
            "titles": ["이종사촌", "이종형제"], "gender": Gender.MALE, "age": AgeGroup.YOUNG_ADULT
        },

        # === 외증조/외방계 ===
        ("나", "어머니", "아버지", "아버지"): {
            "titles": ["증조외할아버지", "외증조부"], "gender": Gender.MALE, "age": AgeGroup.SENIOR
        },
        ("나", "어머니", "아버지", "어머니"): {
            "titles": ["증조외할머니", "외증조모"], "gender": Gender.FEMALE, "age": AgeGroup.SENIOR
        },

        ("나", "어머니", "아버지", "형"): {
            "titles": ["큰외할아버지", "외종조부"], "gender": Gender.MALE, "age": AgeGroup.SENIOR
        },
        ("나", "어머니", "아버지", "형", "아내"): {
            "titles": ["큰외할머니", "외종조모"], "gender": Gender.FEMALE, "age": AgeGroup.SENIOR
        },
        ("나", "어머니", "아버지", "남동생"): {
            "titles": ["작은외할아버지", "외종조부"], "gender": Gender.MALE, "age": AgeGroup.SENIOR
        },
        ("나", "어머니", "아버지", "남동생", "아내"): {
            "titles": ["작은외할머니", "외종조모"], "gender": Gender.FEMALE, "age": AgeGroup.SENIOR
        },
        ("나", "어머니", "아버지", "누나"): {
            "titles": ["고모외할머니", "외대고모"], "gender": Gender.FEMALE, "age": AgeGroup.SENIOR
        },
        ("나", "어머니", "아버지", "누나", "남편"): {
            "titles": ["고모외할아버지", "외대고모부"], "gender": Gender.MALE, "age": AgeGroup.SENIOR
        },
        ("나", "어머니", "아버지", "여동생"): {
            "titles": ["고모외할머니", "외대고모"], "gender": Gender.FEMALE, "age": AgeGroup.SENIOR
        },
        ("나", "어머니", "아버지", "여동생", "남편"): {
            "titles": ["고모외할아버지", "외대고모부"], "gender": Gender.MALE, "age": AgeGroup.SENIOR
        },

        # === 외외가 (어머니의 어머니 쪽) ===
        ("나", "어머니", "어머니", "오빠"): {
            "titles": ["외삼촌할아버지", "외종조부"], "gender": Gender.MALE, "age": AgeGroup.SENIOR
        },
        ("나", "어머니", "어머니", "남동생"): {
            "titles": ["외삼촌할아버지", "외종조부"], "gender": Gender.MALE, "age": AgeGroup.SENIOR
        },
        ("나", "어머니", "어머니", "언니"): {
            "titles": ["큰이모할머니", "이모할머니", "대이모"], "gender": Gender.FEMALE, "age": AgeGroup.SENIOR
        },
        ("나", "어머니", "어머니", "언니", "남편"): {
            "titles": ["이모할아버지", "대이모부"], "gender": Gender.MALE, "age": AgeGroup.SENIOR
        },
        ("나", "어머니", "어머니", "여동생"): {
            "titles": ["작은이모할머니", "이모할머니", "대이모"], "gender": Gender.FEMALE, "age": AgeGroup.SENIOR
        },
        ("나", "어머니", "어머니", "여동생", "남편"): {
            "titles": ["이모할아버지", "대이모부"], "gender": Gender.MALE, "age": AgeGroup.SENIOR
        },

        # === 배우자 관계 ===
        ("나", "남편", "아버지"): {
            "titles": ["시아버지", "아버님"], "gender": Gender.MALE, "age": AgeGroup.ADULT
        },
        ("나", "남편", "어머니"): {
            "titles": ["시어머니", "어머님"], "gender": Gender.FEMALE, "age": AgeGroup.ADULT
        },
        ("나", "남편", "형"): {
            "titles": ["아주버님", "시숙"], "gender": Gender.MALE, "age": AgeGroup.YOUNG_ADULT
        },
        ("나", "남편", "형", "아내"): {
            "titles": "형님", "gender": Gender.FEMALE, "age": AgeGroup.YOUNG_ADULT
        },
        ("나", "남편", "남동생"): {
            "titles": ["도련님", "서방님"], "gender": Gender.MALE, "age": AgeGroup.YOUNG_ADULT
        },
        ("나", "남편", "남동생", "아내"): {
            "titles": "동서", "gender": Gender.FEMALE, "age": AgeGroup.YOUNG_ADULT
        },
        ("나", "남편", "누나"): {
            "titles": ["형님", "시누이"], "gender": Gender.FEMALE, "age": AgeGroup.YOUNG_ADULT
        },
        ("나", "남편", "누나", "남편"): {
            "titles": ["아주버님", "고모부님"], "gender": Gender.MALE, "age": AgeGroup.YOUNG_ADULT
        },
        ("나", "남편", "여동생"): {
            "titles": ["아가씨", "시누이"], "gender": Gender.FEMALE, "age": AgeGroup.YOUNG_ADULT
        },
        ("나", "남편", "여동생", "남편"): {
            "titles": "서방님", "gender": Gender.MALE, "age": AgeGroup.YOUNG_ADULT
        },

        ("나", "아내", "아버지"): {
            "titles": ["장인", "장인어른", "아버님"], "gender": Gender.MALE, "age": AgeGroup.ADULT
        },
        ("나", "아내", "어머니"): {
            "titles": ["장모", "장모님", "어머님"], "gender": Gender.FEMALE, "age": AgeGroup.ADULT
        },
        ("나", "아내", "오빠"): {
            "titles": ["형님", "처남"], "gender": Gender.MALE, "age": AgeGroup.YOUNG_ADULT
        },
        ("나", "아내", "오빠", "아내"): {
            "titles": ["아주머니", "처남댁"], "gender": Gender.FEMALE, "age": AgeGroup.YOUNG_ADULT
        },
        ("나", "아내", "남동생"): {
            "titles": "처남", "gender": Gender.MALE, "age": AgeGroup.YOUNG_ADULT
        },
        ("나", "아내", "남동생", "아내"): {
            "titles": "처남댁", "gender": Gender.FEMALE, "age": AgeGroup.YOUNG_ADULT
        },
        ("나", "아내", "언니"): {
            "titles": "처형", "gender": Gender.FEMALE, "age": AgeGroup.YOUNG_ADULT
        },
        ("나", "아내", "언니", "남편"): {
            "titles": ["형님", "동서"], "gender": Gender.MALE, "age": AgeGroup.YOUNG_ADULT
        },
        ("나", "아내", "여동생"): {
            "titles": "처제", "gender": Gender.FEMALE, "age": AgeGroup.YOUNG_ADULT
        },
        ("나", "아내", "여동생", "남편"): {
            "titles": ["동서", "제부", "서방"], "gender": Gender.MALE, "age": AgeGroup.YOUNG_ADULT
        },

        # === 형제자매의 자녀 (조카) ===
        ("나", "형", "자녀"): {
            "titles": ["조카", "질자"], "gender": Gender.MALE, "age": AgeGroup.CHILD
        },
        ("나", "형", "자녀"): {
            "titles": ["조카", "질녀"], "gender": Gender.FEMALE, "age": AgeGroup.CHILD
        },
        ("나", "남동생", "자녀"): {
            "titles": ["조카", "질자"], "gender": Gender.MALE, "age": AgeGroup.CHILD
        },
        ("나", "남동생", "자녀"): {
            "titles": ["조카", "질녀"], "gender": Gender.FEMALE, "age": AgeGroup.CHILD
        },
        ("나", "누나", "자녀"): {
            "titles": ["조카", "생질", "고종조카"], "gender": Gender.MALE, "age": AgeGroup.CHILD
        },
        ("나", "여동생", "자녀"): {
            "titles": ["조카", "생질", "고종조카"], "gender": Gender.FEMALE, "age": AgeGroup.CHILD
        },
    }

@functools.lru_cache(None)
def get_actors_db():
    return [
        # === [Foreground 1: Left side of table] ===
        {
            "id": "P_FLORAL_TABLE",
            "visual_features": [
                "테이블에 앉아 강아지를 안고 있는 노년의 여성",
                "화려한 꽃무늬 블라우스를 입고 안경을 쓴 여성",
                "흰색 강아지를 품에 안고 있는 여성",
            ],
            "gender": Gender.FEMALE, "age": AgeGroup.SENIOR
        },
        {
            "id": "P_BLUE_TABLE",
            "visual_features": [
                "두 팔을 벌리고 활짝 웃으며 말하고 있는 빨간 모자의 남성",
                "강아지를 안은 노년 여성 바로 옆에 앉아 있는 파란 셔츠의 남성",
                "진한 파란색 셔츠를 입고 빨간색 니트 비니를 쓴 남성",
            ],
            "gender": Gender.MALE, "age": AgeGroup.ADULT
        },
        {
            "id": "P_BLACK_TABLE",
            "visual_features": [
                "파란 셔츠 남성과 줄무늬 상의 남성 사이에 앉아 있는 여성",
                "검은 옷을 입고 풍성한 갈색 곱슬머리를 한 여성",
                "풍성한 갈색 곱슬머리를 한 여성",
            ],
            "gender": Gender.FEMALE, "age": AgeGroup.ADULT
        },
        # === [Foreground 2: Center of table] ===
                {
            "id": "P_WHITE_TABLE",
            "visual_features": [
                "흰색 셔츠를 입고 검은색 선글라스를 착용한 여성",
                "줄무늬 상의의 남성 옆에 서 있는 여성",
                "짧은 금발 숏컷 머리를 한 서 있는 여성",
            ],
            "gender": Gender.FEMALE, "age": AgeGroup.ADULT
        },
        {
            "id": "P_STRIPES_TABLE", 
            "visual_features": [
                "테이블에 앉아 있는 줄무늬 상의의 남성",
                "검은색 야구 모자를 쓰고 수염을 기른 남성",
                "흰색과 검은색 가로 줄무늬 카라 티셔츠를 입은 남성",
            ],
            "gender": Gender.MALE, "age": AgeGroup.ADULT
        },
        # === [Foreground 3: Right side of table] ===
        {
            "id": "P_YELLOW_T_TABLE",
            "visual_features": [
                "테이블에 앉아 있는 노년 남성 무릎 위에 앉은 어린 남성",
                "밝은 노란색 반팔 티셔츠를 입은 어린 남성",
                "갈색 곱슬머리를 한 어린 남성",
            ],
            "gender": Gender.MALE, "age": AgeGroup.CHILD
        },
        {
            "id": "P_GRAY_TABLE",
            "visual_features": [
                "테이블에 앉아 아이를 무릎에 올려둔 노년의 남성",
                "회색 정장 재킷과 흰색 셔츠를 입은 백발의 남성",
                "검은색 나비넥타이를 매고 백발이 성성한 남성",
            ],
            "gender": Gender.MALE, "age": AgeGroup.SENIOR
        },
        {
            "id": "P_YELLOW_TABLE",
            "visual_features": [
                "노년의 남성 옆자리에 앉아 있는 노년의 여성",
                "겨자색 니트 스웨터를 입은 여성",
                "짧은 흰 머리에 노란색 옷을 입은 여성",
            ],
            "gender": Gender.FEMALE, "age": AgeGroup.SENIOR
        },
        {
            "id": "P_YELLOW_DRESS_TABLE",
            "visual_features": [
                "테이블에 앉아 노란 니트 여성 옆에 있는 어린 여성",
                "노란색 원피스를 입고 뒷모습만 보이고 있는 어린 여성",
                "양쪽으로 머리를 묶은 어린 여성",
            ],
            "gender": Gender.FEMALE, "age": AgeGroup.CHILD
        },


        # === [Background: Garden] ===
        # Wine
        {
            "id": "P_GREEN_WINE",
            "visual_features": [
                "짙은 초록색 셔츠와 베이지색 바지를 입은 남성",
                "한 손에 와인잔을 들고 보라색 옷을 입은 여성과 이야기중인 남성",
                "오른손으로 와인을 들고 있는 남성",
            ],
            "gender": Gender.MALE, "age": AgeGroup.ADULT
        },
        {
            "id": "P_PURPLE_WINE",
            "visual_features": [
                "정원 잔디밭에 서 있는 보라색 옷의 여성",
                "진한 보라색 맨투맨 티셔츠를 입은 여성",
                "한 손에 와인잔을 들고 있는 여성",
            ],
            "gender": Gender.FEMALE, "age": AgeGroup.ADULT
        },
        # Grill
        {
            "id": "P_BLUE_GRILL",
            "visual_features": [
                "파란색 티셔츠와 청바지를 입고 고기를 굽는 백발의 남성",
                "고기를 뒤집고있는 백발의 남성",
                "화란색 셔츠를 입은 백발의 남성",
            ],
            "gender": Gender.MALE, "age": AgeGroup.SENIOR
        },
        {
            "id": "P_GRAY_GRILL",
            "visual_features": [
                "그릴에서 고기를 굽고있는 3명의 남성 중 가운데 남성",
                "회색 카라티를 입고 고기를 굽고있는 남성",
                "회색 티셔츠를 입은 그릴 뒤쪽에 있는 남성",
            ],
            "gender": Gender.MALE, "age": AgeGroup.YOUNG_ADULT
        },
        {
            "id": "P_CHECK_GRILL",
            "visual_features": [
                "베이지색 앞치마를 한 남성",
                "빨간 체크무늬 셔츠와 베이지색 앞치마를 착용한 남성",
                "고기를 굽고있는 앞치마를 한 남성",
            ],
            "gender": Gender.MALE, "age": AgeGroup.ADULT
        },

        # Beer
        {
            "id": "P_GREEN_BEER",
            "visual_features": [
                "초록색 후드티를 입은 여성",
                "맥주를 손에들고 남성과 이야기중인 여성",
                "포니테일을 하고 후드티를 입은 여성",
            ],
            "gender": Gender.FEMALE, "age": AgeGroup.YOUNG_ADULT
        },
        {
            "id": "P_BEIGE_BEER",
            "visual_features": [
                "베이지색 조끼를 입은 남성",
                "맥주를 들고 여성과 이야기중인 남성",
                "반팔에 베이지색 조끼를 입은 남성",
            ],
            "gender": Gender.MALE, "age": AgeGroup.YOUNG_ADULT
        },
        # Book
        {
            "id": "P_YELLOW_BOOK",
            "visual_features": [
                "정원에 있는 나무 벤치에 혼자 앉아 있는 여성",
                "노란색 긴팔 니트를 입은 여성",
                "다리를 꼬고 앉아 책을 읽고 있는 여성",
            ],
            "gender": Gender.FEMALE, "age": AgeGroup.YOUNG_ADULT
        },
    ]

def find_best_actor(actors, target_gender, target_age_group, exclude_ids, context_chain=None):
    available = [p for p in actors if p['id'] not in exclude_ids]
    
    matches = [p for p in available if p['gender'] == target_gender and p['age'] == target_age_group]
    
    if not matches:
        logging.warning(
            # f"[Actor Shortage] Chain: {context_chain} | "
            f"Required: {target_gender.value}/{target_age_group.value}"
        )
        return None
    
    return random.choice(matches)

def generate_question():
    title_map = get_relation_chain_to_title()
    actors_db = get_actors_db()

    # 1. Select chain (only chains with length >= 3)
    available_chains = [chain for chain in title_map.keys() if len(chain) >= 3]
    if not available_chains:
        logging.warning("[Chain Selection Failed] No relation chains with length >= 3 available.")
        return generate_question()
    relation_chain = random.choice(available_chains)
    logging.debug(f"[Chain Selected] {relation_chain}")
    
    # 2. Extract answer information (parse dict format)
    relation_info = title_map[relation_chain]
    titles = relation_info["titles"]
    
    if isinstance(titles, list):
        answer = ", ".join(titles)
    else:
        answer = titles

    used_ids = set()

    # 3. Traverse chain (assign actors)
    chain_actors = [] 
    for i in range(2, len(relation_chain) + 1):
        sub_chain = relation_chain[:i]
        rel_key = relation_chain[i-1]
        
        step_info = title_map.get(sub_chain)
        
        if step_info:
            target_gender = step_info["gender"]
            target_age = step_info["age"]
            
            raw_title = step_info["titles"]
            title_str = raw_title[0] if isinstance(raw_title, list) else raw_title
        else:
            logging.warning(f"[Relation Info Missing] Chain: {sub_chain}")
            return generate_question()

        actor = find_best_actor(actors_db, target_gender, target_age, used_ids, context_chain=sub_chain)
        
        if not actor:
            return generate_question()
        
        used_ids.add(actor['id'])
        chain_actors.append((actor, title_str, rel_key))
    
    target_actor, _, target_rel_key = chain_actors[-1]
    
    # 5. Set bridge actor
    bridge_actor = None
    if len(chain_actors) >= 2:
        bridge_actor, _, _ = chain_actors[-2]

    # 6. Construct question (dialogue format)
    dialogue_map = get_dialogue_templates()
    relative_names = get_friend_names()
    
    # Generate person map (use actor's visual_features as person descriptors)
    person_map = {}
    choice_feature = None  # Feature for last actor's choice
    
    for i, (actor, title_str, rel_key) in enumerate(chain_actors, 1):
        # For last actor: randomly select 2 out of 3 visual_features
        if i == len(chain_actors):
            selected_features = random.sample(actor['visual_features'], min(2, len(actor['visual_features'])))
            person_map[i] = selected_features[0]  # Used in dialogue
            choice_feature = selected_features[1]  # Used in choices
        else:
            person_map[i] = random.choice(actor['visual_features'])
    
    dialogue_lines = []
    
    intro_sentences = [
        "오랜만에 친구들이 우리 집에 놀러 왔다. 거실 벽에 걸린 가족사진을 보며 친구들이 이것저것 물어보았다.\n",
        "집들이에 친구들을 초대했다. 새로 꾸민 거실을 구경하던 친구가 벽에 걸린 가족사진을 발견하고 물었다.\n",
        "동창회에서 만난 친구와 카페에 앉아 이야기를 나누다가, 휴대폰 속 가족사진을 보여주며 이야기를 꺼냈다.\n"
    ]
    
    dialogue_lines.append(random.choice(intro_sentences))
    
    used_speakers = set()
    
    for i in range(len(relation_chain) - 1):
        source_rel = relation_chain[i]
        target_rel = relation_chain[i+1]
        
        clue_key = (source_rel, target_rel)
        
        if clue_key not in dialogue_map:
            logging.warning(f"Dialogue template not found: {clue_key} (source_rel: {source_rel}, target_rel: {target_rel})")
            return generate_question()
        
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
            
            if target_placeholder:
                last_char = target_placeholder[-1]
                if '가' <= last_char <= '힣':
                    has_final = (ord(last_char) - 0xAC00) % 28 != 0
                    correct_nominative = "이" if has_final else "가"
                    dialogue_line = dialogue_line.replace(f"{target_placeholder}이 ", f"{target_placeholder}{correct_nominative} ")
            
            dialogue_lines.append(dialogue_line)
    
    last_index = len(relation_chain) - 1
    
    available_for_wrong = [p for p in actors_db if p['id'] != target_actor['id']]
    wrong_actors = random.sample(available_for_wrong, 3)
    
    all_options = [target_actor] + wrong_actors
    random.shuffle(all_options)
    
    choices = {}
    correct_letter = None
    for i, actor in enumerate(all_options):
        letter = chr(65 + i)  # A, B, C, D
        feature = random.choice(actor['visual_features'])
        
        if actor['id'] == target_actor['id']:
            correct_letter = letter
            feature = choice_feature if choice_feature else random.choice(actor['visual_features'])
        
        choices[letter] = feature
    
    if isinstance(titles, list):
        selected_title = random.choice(titles)
    else:
        selected_title = titles
    
    particle = get_proper_particle(selected_title)
    dialogue_lines.append(f"\n이때, 나의 {selected_title}{particle} 누구인가?")
    for letter in ['A', 'B', 'C', 'D']:
        dialogue_lines.append(f"{letter}: {choices[letter]}")
    
    question = "\n".join(dialogue_lines)
    
    # 8. Explanation
    explanation = ["[STEP 0] Interpret the given dialogue to identify relationships between people."]
    temp_chain_str = "나"
    
    for i, rel in enumerate(relation_chain[1:], 1):
        person = person_map[i]
        explanation.append(f"[STEP {i}] Through the dialogue, infer that '{person}' is '{temp_chain_str}의 {rel}' relationship.")
        temp_chain_str += f"의 {rel}"
    
    final_step = len(relation_chain)
    if choice_feature and choice_feature != person_map[last_index]:
        explanation.append(f"[STEP {final_step}] Confirm that '{person_map[last_index]}' and '{choice_feature}' are the same person.")
        explanation.append(f"[STEP {final_step + 1}] Therefore, the final title for the combined relationship '{temp_chain_str}' is '{answer}'.")
    else:
        explanation.append(f"[STEP {final_step}] Therefore, the final title for the combined relationship '{temp_chain_str}' is '{answer}'.")
    
    explanation.append(f"[Answer] {correct_letter}: {choices[correct_letter]}")

    return question, correct_letter, explanation, choices

def create_dataset_files(num_questions, version):
    print(f"Generating {num_questions} kinship problems (Visual Benchmark)...")
    
    actors_db = get_actors_db()
    title_map = get_relation_chain_to_title()
    
    print(f"\n=== Data Statistics ===")
    print(f"Total relation chains: {len(title_map)}")
    print(f"Total actors: {len(actors_db)}")
    
    gender_age_counts = {}
    for actor in actors_db:
        key = f"{actor['gender'].value}/{actor['age'].value}"
        gender_age_counts[key] = gender_age_counts.get(key, 0) + 1
    
    print("Actor distribution:")
    for combo, count in sorted(gender_age_counts.items()):
        print(f"  {combo}: {count} people")
    print("=" * 40 + "\n")
    
    output = []
    
    for i in range(num_questions):
        q, a, e, choices = generate_question()
        output.append([q, a, "\n".join(e), choices['A'], choices['B'], choices['C'], choices['D']])
    
    df = pd.DataFrame(output, columns=['question', 'answer', 'solution', 'choice_A', 'choice_B', 'choice_C', 'choice_D'])
    print(f"Total: {len(df)}, Unique: {df['question'].nunique()}")
    
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    
    csv_path = PROJECT_ROOT / "data" / "csv" / f"KINSHIP_VISION_{version}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"CSV saved: {csv_path}")
    
    json_path = PROJECT_ROOT / "data" / "json" / f"KINSHIP_VISION_{version}.jsonl"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    
    json_data = [
        {
            "question": r['question'], 
            "answer": r['answer'], 
            "solution": r['solution'],
            "choices": {
                "A": r['choice_A'],
                "B": r['choice_B'],
                "C": r['choice_C'],
                "D": r['choice_D']
            }
        } 
        for _, r in df.iterrows()
    ]
    
    with open(json_path, 'w', encoding='utf-8') as f:
        for item in json_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"JSONL saved: {json_path}")
    
    return df, json_data

if __name__ == '__main__':
    kinship_df, kinship_json = create_dataset_files(num_questions=10, version="v5_hard")
    
    print("\n=== Sample Problem ===")
    for _ in range(1):
        question, answer, explanation, choices = generate_question()
        print("Q:", question)
        print("A:", answer)
        print("\n--- Explanation ---")
        for step in explanation:
            print(step)
        print("-" * 40)