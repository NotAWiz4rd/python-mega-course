todos = []

while True:
    user_action = input("Type add, or show, or exit: ").strip().lower()

    match user_action:
        case 'add':
            todo = input("Enter a todo: ")
            todos.append(todo)
        case 'show':
            for todo in todos:
                print(todo.title())
        case 'exit':
            break
        case whatever:
            print(f"'{whatever}' is not a valid action")

print("Goodbye!")
