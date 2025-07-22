from jarvis_cd.basic.pkg import Application
from jarvis_util.shell.exec import Exec
from jarvis_util.shell.pssh_exec import PsshExecInfo
from jarvis_util import *
import os

class LightbeamDistributedTest(Application):
    def _init(self):
        pass

    def _configure_menu(self):
        return [
            {'name': 'hostfile', 'msg': 'Path to hostfile', 'type': str, 'default': None},
            {'name': 'transport', 'msg': 'Transport (zeromq or thallium)', 'type': str, 'default': 'thallium'},
            {'name': 'protocol', 'msg': 'Protocol (tcp or ofi+sockets)', 'type': str, 'default': 'ofi+sockets'},
            {'name': 'domain', 'msg': 'Domain (e.g., localhost)', 'type': str, 'default': 'localhost'},
            {'name': 'port', 'msg': 'Port (e.g., 8200)', 'type': int, 'default': 8200}
        ]

    def _configure(self, **kwargs):
        self.update_config(kwargs, rebuild=False)
        self.hostfile = self.config['hostfile']
        self.transport = self.config['transport']
        self.protocol = self.config['protocol']
        self.domain = self.config['domain']
        self.port = self.config['port']

    def start(self):
        self.binary = 'distributed_lightbeam_test'
        # Replace None or empty string with '' for all parameters before use
        def quote_if_empty(val):
            return "''" if val is None or (isinstance(val, str) and val.strip() == '') else str(val)
        self.transport = quote_if_empty(self.config['transport'])
        self.hostfile = quote_if_empty(self.config['hostfile'])
        self.protocol = quote_if_empty(self.config['protocol'])
        self.domain = quote_if_empty(self.config['domain'])
        self.port = quote_if_empty(self.config['port'])
        cmd = f"{self.binary} {self.transport} {self.hostfile} {self.protocol} {self.domain} {self.port}"
        print(f"[LightbeamDistributedTest] Launching: {cmd}")
        hosts = Hostfile(path=self.config['hostfile'])
        print(f"[LightbeamDistributedTest] Hosts: {hosts} , len(hosts): {len(hosts)}", hosts.path)
        print(f" cmd: {cmd}")
        # hosts = self.jarvis.hostfile
        Exec(cmd, MpiExecInfo(hosts= hosts, env=self.env, exec_async=False, nprocs=len(hosts), ppn=2))

    def stop(self):
        """
        Stop a running application. E.g., terminate servers, clients, etc.
        """
        pass

    def clean(self):
        """
        Destroy all data for the application. E.g., delete metadata, data directories, etc.
        """
        pass 