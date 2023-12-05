class Task:
    def __init__(self, title, date, description):
        self.title = title
        self.date = date
        self.description = description

class TaskManager:
    def __init__(self):
        self.tasks = []

    def add_task(self, title, date, description):
        task = Task(title, date, description)
        self.tasks.append(task)
        print(f'Task "{title}" added successfully.')

    def view_tasks(self):
        if not self.tasks:
            print('No tasks available.')
        else:
            print('\nTask List:')
            for idx, task in enumerate(self.tasks, start=1):
                print(f'{idx}. Title: {task.title}\n   Date: {task.date}\n   Description: {task.description}\n')

def main():
    task_manager = TaskManager()

    while True:
        print('\n=== Task Management System ===')
        print('1. Add Task')
        print('2. View Tasks')
        print('3. Exit')

        choice = input('Enter your choice (1/2/3): ')

        if choice == '1':
            title = input('Enter task title: ')
            date = input('Enter task date: ')
            description = input('Enter task description: ')
            task_manager.add_task(title, date, description)
        elif choice == '2':
            task_manager.view_tasks()
        elif choice == '3':
            print('Exiting Task Management System. Goodbye!')
            break
        else:
            print('Invalid choice. Please enter a valid option.')

if __name__ == "__main__":
    main()
