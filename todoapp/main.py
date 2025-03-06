todos = []

while True:
    user_action = input("Type add, or show, edit or exit: ").strip().lower()

    match user_action:
        case 'add':
            todo = input("Enter a todo: ")
            todos.append(todo)
        case 'show':
            for todo in todos:
                print(todo.title())
        case 'edit':
            number = int(input("Number of the todo to edit: "))
            existing_todo_index = number - 1
            new_todo = input("Enter new todo: ")
            todos[existing_todo_index] = new_todo
        case 'exit':
            break
        case whatever:
            print(f"'{whatever}' is not a valid action")

print("Goodbye!")
