from inference import generate_code
import distro


distro_info = distro.id()
print("Hello! My name is Askterm, you can use me for linux commands\n" 
"exit - 0")
while True:
     try:
       text = input('install me: ') 
       prompt = f'How to install {text} on Linux/{distro_info}'
       response = generate_code(prompt)
       print(response)
     except Exception as e:
         print(e.add_note)
       