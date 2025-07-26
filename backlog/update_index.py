#!/usr/bin/env python
"""
Update the backlog index.md file based on the current task files.

This script scans all task files in the backlog directory structure,
extracts their metadata, and generates an updated index.md file.
"""

import os
import re
import yaml
from datetime import datetime
from pathlib import Path

# Categories to organize backlog items
CATEGORIES = ['documentation', 'infrastructure', 'features', 'bugs']
STATUSES = ['proposed', 'ready', 'in_progress', 'completed', 'abandoned']

def extract_task_metadata(filepath):
    """Extract metadata from a task file."""
    # Extract category from filepath
    filepath_str = str(filepath)
    category = None
    for cat in CATEGORIES:
        if f"{os.sep}{cat}{os.sep}" in filepath_str:
            category = cat
            break
    
    # Extract task ID from filename
    task_id = filepath.stem
    
    metadata = {
        'filepath': filepath,
        'id': task_id,
        'category': category,
        'title': None,
        'status': None,
        'priority': None,
        'created': None,
        'updated': None
    }
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for YAML frontmatter
        if content.startswith('---'):
            # Split content to extract YAML frontmatter
            parts = content.split('---', 2)
            if len(parts) >= 3:
                yaml_content = parts[1].strip()
                try:
                    yaml_data = yaml.safe_load(yaml_content)
                    if yaml_data:
                        # Extract fields from YAML
                        metadata['title'] = yaml_data.get('title', '').strip('"')
                        status = yaml_data.get('status', '').strip('"').lower().replace(' ', '_')
                        metadata['status'] = status if status else None
                        metadata['priority'] = yaml_data.get('priority', '').strip('"')
                        metadata['created'] = yaml_data.get('created', '').strip('"')
                        # Handle both 'updated' and 'last_updated' fields
                        updated = yaml_data.get('updated', yaml_data.get('last_updated', '')).strip('"')
                        metadata['updated'] = updated if updated else None
                        return metadata
                except yaml.YAMLError as e:
                    print(f"YAML parsing error in {filepath}: {e}")
        
        # Fallback to markdown header extraction if no valid YAML
        title_match = re.search(r'^# (?:Task: |Bug: )?(.*)', content, re.MULTILINE)
        if title_match:
            metadata['title'] = title_match.group(1).strip()
        
        # Try to extract metadata from old format
        if 'Status:' in content:
            # Extract status
            status_match = re.search(r'\*\*Status\*\*: (.*)', content)
            if status_match:
                status = status_match.group(1).strip().lower().replace(' ', '_')
                metadata['status'] = status
            
            # Extract priority
            priority_match = re.search(r'\*\*Priority\*\*: (.*)', content)
            if priority_match:
                metadata['priority'] = priority_match.group(1).strip()
            
            # Extract created date
            created_match = re.search(r'\*\*Created\*\*: (.*)', content)
            if created_match:
                metadata['created'] = created_match.group(1).strip()
            
            # Extract updated date
            updated_match = re.search(r'\*\*Last Updated\*\*: (.*)', content)
            if updated_match:
                metadata['updated'] = updated_match.group(1).strip()
                
        return metadata
    except Exception as e:
        print(f"Error processing {filepath}: {str(e)}")
        return None

def find_all_task_files():
    """Find all task files in the backlog directory."""
    backlog_dir = Path(__file__).parent
    task_files = []
    
    print(f"Searching for task files in {backlog_dir}")
    
    for category in CATEGORIES:
        category_dir = backlog_dir / category
        if category_dir.exists():
            print(f"Checking category directory: {category_dir}")
            for file_path in category_dir.glob('*.md'):
                # Skip template files and README files
                if file_path.name.startswith('template') or file_path.name.lower() == 'readme.md':
                    continue
                print(f"Found task file: {file_path}")
                task_files.append(file_path)
    
    return task_files

def filter_valid_tasks(task_metadata_list):
    """Filter out tasks that don't have required metadata."""
    valid_tasks = []
    for task in task_metadata_list:
        if task is None:
            continue
        
        # Check for required fields
        if not task.get('title') or not task.get('status'):
            print(f"Skipping invalid task: {task}")
            continue
        
        # Normalize status to lowercase with underscores
        if task.get('status'):
            task['status'] = task['status'].lower().replace(' ', '_')
        
        valid_tasks.append(task)
    
    return valid_tasks

def sort_tasks_by_priority_and_date(tasks):
    """Sort tasks by priority (high first) and then by creation date."""
    priority_order = {'high': 0, 'medium': 1, 'low': 2}
    
    def sort_key(task):
        priority = task.get('priority', 'medium').lower()
        priority_value = priority_order.get(priority, 1)
        
        # Use created date for secondary sort, handle None values
        created_date = task.get('created', '')
        if not created_date:
            created_date = '9999-12-31'  # Put tasks without dates at the end
        
        return (priority_value, created_date)
    
    return sorted(tasks, key=sort_key)

def generate_index_content(tasks):
    """Generate the markdown content for the index file."""
    content = [
        "# Backlog Index",
        "",
        f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
        f"Total tasks: {len(tasks)}",
        ""
    ]
    
    # Group tasks by status
    status_groups = {}
    for task in tasks:
        status = task.get('status', 'unknown')
        if status not in status_groups:
            status_groups[status] = []
        status_groups[status].append(task)
    
    # Generate sections for each status
    for status in STATUSES:
        if status in status_groups:
            tasks_in_status = status_groups[status]
            status_title = status.replace('_', ' ').title()
            content.extend([
                f"## {status_title} ({len(tasks_in_status)})",
                ""
            ])
            
            # Group by category within status
            category_groups = {}
            for task in tasks_in_status:
                category = task.get('category', 'unknown')
                if category not in category_groups:
                    category_groups[category] = []
                category_groups[category].append(task)
            
            for category in CATEGORIES:
                if category in category_groups:
                    category_tasks = category_groups[category]
                    content.extend([
                        f"### {category.title()}",
                        ""
                    ])
                    
                    for task in category_tasks:
                        title = task.get('title', 'Untitled')
                        priority = task.get('priority', 'medium')
                        created = task.get('created', 'unknown')
                        updated = task.get('updated', 'unknown')
                        
                        priority_emoji = {'high': '🔥', 'medium': '📝', 'low': '💡'}.get(priority, '📝')
                        
                        content.append(f"- {priority_emoji} **{title}** (Priority: {priority})")
                        content.append(f"  - ID: `{task['id']}`")
                        content.append(f"  - Created: {created}")
                        content.append(f"  - Updated: {updated}")
                        content.append("")
            
            content.append("")
    
    return '\n'.join(content)

def main():
    """Main function to update the backlog index."""
    # Find all task files
    task_files = find_all_task_files()
    
    if not task_files:
        print("No task files found.")
        return
    
    # Extract metadata from all files
    task_metadata_list = [extract_task_metadata(filepath) for filepath in task_files]
    
    # Filter valid tasks
    valid_tasks = filter_valid_tasks(task_metadata_list)
    
    if not valid_tasks:
        print("No valid tasks found.")
        return
    
    # Sort tasks
    sorted_tasks = sort_tasks_by_priority_and_date(valid_tasks)
    
    # Generate index content
    index_content = generate_index_content(sorted_tasks)
    
    # Write to index.md
    index_path = Path(__file__).parent / 'index.md'
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(index_content)
    
    print(f"Updated {index_path} with {len(valid_tasks)} tasks.")

if __name__ == '__main__':
    main() 