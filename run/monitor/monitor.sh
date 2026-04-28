#!/bin/bash

# Evaluation monitoring script for logical-puzzles
# Works on macOS, Ubuntu, Git Bash
# Mode selection: simple or detailed (default: simple)
MODE=${1:-simple}

# Color definitions
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Get project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$PROJECT_ROOT/results/log"

# Common function: extract argument from command line
extract_arg() {
    local line="$1"
    local arg_name="$2"
    echo "$line" | grep -oE -- "--${arg_name} [^ ]+" | awk '{print $2}' | head -n 1
}

# Common function: extract progress from log file
extract_progress() {
    local log_file="$1"
    
    if [ ! -f "$log_file" ]; then
        echo "N/A"
        return
    fi
    
    # Search last 50 lines for progress information
    local progress_line=$(tail -n 50 "$log_file" | grep -E "API calls progress:|Processing completed:|Accuracy:" | tail -n 1)
    
    if [ -z "$progress_line" ]; then
        # Check if task is completed
        if grep -q "Status: SUCCESS\|Status: FAILED" "$log_file" 2>/dev/null; then
            if grep -q "Status: SUCCESS" "$log_file" 2>/dev/null; then
                echo "Completed ✓"
            else
                echo "Failed ✗"
            fi
        else
            echo "Starting..."
        fi
        return
    fi
    
    # Extract progress percentage
    local progress_percent=$(echo "$progress_line" | grep -oE '[0-9]{1,3}%' | head -n 1)
    if [ -n "$progress_percent" ]; then
        echo "$progress_percent"
        return
    fi
    
    # Extract completed count
    local completed=$(echo "$progress_line" | grep -oE '[0-9]+/[0-9]+' | head -n 1)
    if [ -n "$completed" ]; then
        echo "$completed"
        return
    fi
    
    echo "In Progress"
}

# Extract accuracy from log
extract_accuracy() {
    local log_file="$1"
    
    if [ ! -f "$log_file" ]; then
        echo "N/A"
        return
    fi
    
    # Search for accuracy line
    local accuracy_line=$(grep -E "Accuracy:" "$log_file" | tail -n 1)
    
    if [ -n "$accuracy_line" ]; then
        local accuracy=$(echo "$accuracy_line" | grep -oE '[0-9]+\.[0-9]+%' | head -n 1)
        if [ -n "$accuracy" ]; then
            echo "$accuracy"
            return
        fi
    fi
    
    echo "N/A"
}

if [ "$MODE" = "detailed" ] || [ "$MODE" = "d" ]; then
    # Find running evaluation processes
    PROCESS_LIST=$(ps -ax -o command 2>/dev/null | grep '[e]valuation/run.py' || ps -eo command 2>/dev/null | grep '[e]valuation/run.py')
    
    if [ -z "$PROCESS_LIST" ]; then
        echo -e "${RED}No evaluation currently running.${NC}"
        exit 0
    fi
    
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}🔍 DETAILED MONITORING MODE${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    ps -ax -o pid,command 2>/dev/null | grep '[e]valuation/run.py' | while read -r pid line; do
        # Extract arguments
        model=$(extract_arg "$line" "model")
        task=$(extract_arg "$line" "tasks")
        
        # Default model if not specified
        [ -z "$model" ] && model="default (from config.yaml)"
        
        # Log file path
        log_file=""
        if [ -n "$task" ]; then
            log_file="$LOG_DIR/${task}.log"
        fi
        
        echo -e "${BLUE}─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────${NC}"
        printf "${YELLOW}🆔 %-10s${NC}| %s\n" "PID" "$pid"
        printf "${YELLOW}▶️  %-10s${NC}| %s\n" "Model" "$model"
        printf "${YELLOW}▶️  %-10s${NC}| %s\n" "Task" "$task"
        
        if [ -z "$log_file" ]; then
            printf "${RED}❌ %-10s${NC}| %s\n" "Error" "Cannot determine log file path (task not found)."
            echo -e "${BLUE}─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────${NC}"
            echo ""
            continue
        fi
        
        if [ -f "$log_file" ]; then
            # Extract progress
            progress=$(extract_progress "$log_file")
            accuracy=$(extract_accuracy "$log_file")
            
            printf "${GREEN}📊 %-10s${NC}| %s\n" "Progress" "$progress"
            if [ "$accuracy" != "N/A" ]; then
                printf "${GREEN}📊 %-10s${NC}| %s\n" "Accuracy" "$accuracy"
            fi
            
            # Check status
            if grep -q "Status: SUCCESS" "$log_file" 2>/dev/null; then
                printf "${GREEN}✅ %-10s${NC}| %s\n" "Status" "SUCCESS"
            elif grep -q "Status: FAILED" "$log_file" 2>/dev/null; then
                printf "${RED}❌ %-10s${NC}| %s\n" "Status" "FAILED"
            else
                printf "${YELLOW}⏳ %-10s${NC}| %s\n" "Status" "Running"
            fi
            
            # Start time from log
            start_time=$(grep "Started at:" "$log_file" 2>/dev/null | head -n 1 | sed 's/.*Started at: //')
            if [ -n "$start_time" ]; then
                printf "${BLUE}🕐 %-10s${NC}| %s\n" "Started" "$start_time"
            fi
            
            printf "${BLUE}📁 %-10s${NC}| %s\n" "Log" "$log_file"
        else
            printf "${YELLOW}📁 %-10s${NC}| %s (not created yet)\n" "Log" "$log_file"
        fi
        echo ""
    done
    
elif [ "$MODE" = "simple" ] || [ "$MODE" = "s" ]; then
    # Find running evaluation processes
    PROCESS_LIST=$(ps -ax -o command 2>/dev/null | grep '[e]valuation/run.py' || ps -eo command 2>/dev/null | grep '[e]valuation/run.py')
    
    if [ -z "$PROCESS_LIST" ]; then
        echo -e "${RED}No evaluation currently running.${NC}"
        exit 0
    fi
    
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}📋 SIMPLE TABLE MODE${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    printf "%s\n" "─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────"
    printf "${YELLOW}%-4s | %-7s | %-10s | %-25s | %-20s | %-15s | %s${NC}\n" "No." "PID" "Started" "Model" "Task" "Progress" "Accuracy"
    printf "%s\n" "─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────"
    
    count=0
    ps -ax -o pid,command 2>/dev/null | grep '[e]valuation/run.py' | while read -r pid line; do
        # Extract arguments
        model=$(extract_arg "$line" "model")
        task=$(extract_arg "$line" "tasks")
        
        # Default model if not specified
        [ -z "$model" ] && model="default"
        
        # Start time
        stime=$(ps -p "$pid" -o lstart= 2>/dev/null | awk '{print $4}' || echo "N/A")
        [ -z "$stime" ] && stime="N/A"
        
        count=$((count + 1))
        
        # Progress and accuracy
        log_file=""
        progress="N/A"
        accuracy="N/A"
        
        if [ -n "$task" ]; then
            log_file="$LOG_DIR/${task}.log"
            if [ -f "$log_file" ]; then
                progress=$(extract_progress "$log_file")
                accuracy=$(extract_accuracy "$log_file")
            fi
        fi
        
        printf "%-4s | %-7s | %-10s | %-25s | %-20s | %-15s | %s\n" "$count" "$pid" "$stime" "$model" "$task" "$progress" "$accuracy"
    done
    
    echo ""
    
elif [ "$MODE" = "help" ] || [ "$MODE" = "h" ]; then
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}🔧 Evaluation Monitoring Script${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "${YELLOW}Usage:${NC}"
    echo "  bash scripts/monitor.sh [mode]    # or ./scripts/monitor.sh [mode]"
    echo ""
    echo -e "${YELLOW}Modes:${NC}"
    echo -e "  ${GREEN}simple${NC}, ${GREEN}s${NC}    - Simple table view with progress (default)"
    echo -e "  ${GREEN}detailed${NC}, ${GREEN}d${NC}  - Detailed progress view with full information"
    echo -e "  ${GREEN}help${NC}, ${GREEN}h${NC}      - Show this help message"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo "  bash scripts/monitor.sh              # Simple table"
    echo "  bash scripts/monitor.sh detailed     # Detailed view"
    echo "  bash scripts/monitor.sh d           # Same as detailed"
    echo ""
    echo -e "${YELLOW}Log Location:${NC}"
    echo "  $LOG_DIR"
    echo ""
    echo -e "${YELLOW}Platform Support:${NC}"
    echo "  macOS, Ubuntu, Git Bash (Windows)"
    echo ""
else
    echo -e "${RED}Unknown mode: $MODE${NC}"
    echo "Use 'help' or 'h' for usage information."
    exit 1
fi

# bash scripts/monitor.sh