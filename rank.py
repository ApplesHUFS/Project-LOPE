"""PER 기반 순위 시스템."""

import json
import os
from typing import List, Dict
from datetime import datetime


def load_all_results(results_dir: str = "results") -> List[Dict]:
    """results 폴더의 모든 결과 파일을 로드.
    
    Args:
        results_dir: 결과 파일이 저장된 디렉토리
        
    Returns:
        결과 데이터 리스트
    """
    if not os.path.exists(results_dir):
        print(f"Warning: Results directory '{results_dir}' does not exist")
        return []
    
    results = []
    for filename in os.listdir(results_dir):
        if filename.endswith('.json') and filename != 'rankings.json':
            filepath = os.path.join(results_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    data['result_filename'] = filename
                    results.append(data)
            except Exception as e:
                print(f"Warning: Could not load {filename}: {e}")
                continue
    
    return results


def calculate_rankings(results: List[Dict]) -> List[Dict]:
    """PER 기준으로 순위를 계산 (낮은 PER이 높은 순위).
    
    Args:
        results: 결과 데이터 리스트
        
    Returns:
        순위가 매겨진 결과 리스트
    """
    if not results:
        return []
    
    # PER 기준으로 정렬 (오름차순 - 낮을수록 1등)
    sorted_results = sorted(results, key=lambda x: x['evaluation']['per'])
    
    # 순위 정보 추가
    rankings = []
    current_rank = 1
    prev_per = None
    
    for idx, result in enumerate(sorted_results):
        per_value = result['evaluation']['per']
        
        # 동점자 처리: 같은 PER이면 같은 순위
        if prev_per is not None and per_value != prev_per:
            current_rank = idx + 1
        
        rank_info = {
            'rank': current_rank,
            'participant': result.get('audio_filename', 'Unknown'),
            'per': per_value,
            'per_percentage': result['evaluation']['per_percentage'],
            'rating': result['evaluation']['rating'],
            'timestamp': result.get('timestamp', 'Unknown'),
            'result_file': result.get('result_filename', 'Unknown'),
            'reference_text': result.get('reference_text', ''),
            'canonical_count': result['canonical']['count'],
            'predicted_count': result['predicted']['count']
        }
        
        rankings.append(rank_info)
        prev_per = per_value
    
    return rankings


def save_rankings(rankings: List[Dict], output_path: str = "results/rankings.json"):
    """순위 정보를 JSON 파일로 저장.
    
    Args:
        rankings: 순위 데이터 리스트
        output_path: 저장할 파일 경로
    """
    # results 디렉토리 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 순위 데이터 구성
    ranking_data = {
        'updated_at': datetime.now().isoformat(),
        'total_participants': len(rankings),
        'rankings': rankings
    }
    
    # JSON 파일로 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(ranking_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Rankings saved to: {output_path}")


def display_rankings(rankings: List[Dict]):
    """순위를 콘솔에 표시.
    
    Args:
        rankings: 순위 데이터 리스트
    """
    if not rankings:
        print("\n📊 No results available yet.")
        return
    
    print("\n" + "="*80)
    print("                       🏆 PRONUNCIATION RANKINGS 🏆")
    print("="*80)
    print(f"{'Rank':<6} {'Participant':<25} {'PER':<10} {'Rating':<20} {'Date':<20}")
    print("-"*80)
    
    for rank_info in rankings:
        rank = rank_info['rank']
        participant = rank_info['participant']
        per_pct = rank_info['per_percentage']
        rating = rank_info['rating']
        timestamp = rank_info['timestamp'][:10] if len(rank_info['timestamp']) >= 10 else rank_info['timestamp']
        
        # 1-3등에 메달 표시
        if rank == 1:
            rank_display = "🥇 1"
        elif rank == 2:
            rank_display = "🥈 2"
        elif rank == 3:
            rank_display = "🥉 3"
        else:
            rank_display = f"   {rank}"
        
        print(f"{rank_display:<6} {participant:<25} {per_pct:>5.2f}%     {rating:<20} {timestamp:<20}")
    
    print("="*80)
    print(f"Total participants: {len(rankings)}")
    print("="*80 + "\n")


def update_rankings(results_dir: str = "results", output_path: str = "results/rankings.json"):
    """결과 파일들을 읽어서 순위를 업데이트.
    
    Args:
        results_dir: 결과 파일이 저장된 디렉토리
        output_path: 순위 파일 저장 경로
    """
    print("🔄 Updating rankings...")
    
    # 모든 결과 로드
    results = load_all_results(results_dir)
    
    if not results:
        print("⚠️  No results found to rank.")
        return
    
    # 순위 계산
    rankings = calculate_rankings(results)
    
    # 순위 저장
    save_rankings(rankings, output_path)
    
    # 순위 표시
    display_rankings(rankings)


def main():
    """메인 함수."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Update and display PER-based rankings")
    parser.add_argument(
        '--results_dir',
        type=str,
        default='results',
        help='Directory containing result JSON files'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/rankings.json',
        help='Output path for rankings file'
    )
    parser.add_argument(
        '--display_only',
        action='store_true',
        help='Only display existing rankings without updating'
    )
    
    args = parser.parse_args()
    
    if args.display_only:
        # 기존 rankings.json 파일 읽어서 표시
        if os.path.exists(args.output):
            with open(args.output, 'r', encoding='utf-8') as f:
                ranking_data = json.load(f)
                display_rankings(ranking_data.get('rankings', []))
        else:
            print(f"⚠️  Rankings file not found: {args.output}")
    else:
        # 순위 업데이트
        update_rankings(args.results_dir, args.output)


if __name__ == '__main__':
    main()
