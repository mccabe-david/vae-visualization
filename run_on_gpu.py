import paramiko
import os

from secrets import uname, password

router_ip = "cscigpu02.bc.edu"
router_uname = uname
router_pword = password

ssh = paramiko.SSHClient()

ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

ssh.connect(
    router_ip,
    username=router_uname,
    password=router_pword,
    look_for_keys=False
)

# for copying files to gpu
# (You'll need to enter password)
# os.system("scp vae.py " + uname + router_ip + ":")

# for running python files
# command = "python3 vae.py"

# for package installation
command2 = "pip uninstall -r requirements.txt"

stdin, stdout, sterr = ssh.exec_command(command2)

print(stdout.read().decode())
print(sterr.read().decode())

ssh.close()