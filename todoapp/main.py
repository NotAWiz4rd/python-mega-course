prompt = "Enter todo: "

todos = []

while True:
    todo = input(prompt)
    todo.capitalize()
    todos.append(todo.capitalize())
    print(todos)

