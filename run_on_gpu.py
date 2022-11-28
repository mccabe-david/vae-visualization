import paramiko
import os

from gpu_secrets import uname, password

router_ip = "cscigpu07.bc.edu"
router_uname = uname
router_pword = password

with paramiko.SSHClient() as ssh:

    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    ssh.connect(
        router_ip,
        username=router_uname,
        password=router_pword,
        look_for_keys=False
    )

    # for copying files to gpu
    # (You'll need to enter password)
    os.system("scp vae.py " + uname + "@" + router_ip + ":")
    # os.system("scp requirements.txt " + uname + "@" + router_ip + ":")

    # for running python files
    # cleanup0 = "ls"
    # cleanup1 = "rm *"
    # command = "python3 vae.py"
    command = "python3 vae.py ; curl --upload-file ./output11.png https://salad.keep.sh"

    # for package installation
    # command2 = "pip3 install -r requirements.txt"
    stdin, stdout, sterr = ssh.exec_command(command)

    print(stdout.read().decode())
    print(sterr.read().decode())
    '''
    stdin, stdout, sterr = ssh.exec_command(command)

    print(stdout.read().decode())
    print(sterr.read().decode())

    
    stdin, stdout, sterr = ssh.exec_command(command)

    print(stdout.read().decode())
    print(sterr.read().decode())
    '''
    # os.system("scp mccabedi@cscigpu06:/home/mccabedi/output.png C:\\Users\\dmcca\\OneDrive\\Desktop\\gpu")