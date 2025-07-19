"""
LightBeam Server Package for Jarvis
"""
from jarvis_cd.basic.pkg import Application
from jarvis_util import *


class LightbeamServer(Application):
    def _init(self):
        pass

    def _configure_menu(self):
        return [
            {
                'name': 'SERVER_IP',
                'msg': 'IP address to bind to',
                'type': str,
                'default': '0.0.0.0',
            },
            {
                'name': 'SERVER_PORT',
                'msg': 'Port to bind to',
                'type': int,
                'default': 5555,
            },
            {
                'name': 'TRANSPORT_TYPE',
                'msg': 'Transport type (tcp, libfabric)',
                'type': str,
                'default': 'tcp',
                'choices': ['tcp', 'libfabric']
            },
            {
                'name': 'VERBOSE',
                'msg': 'Enable verbose output',
                'type': bool,
                'default': True,
            },
        ]

    def _configure(self, **kwargs):
        pass

    def start(self):
        if self.config['TRANSPORT_TYPE'] == 'tcp':
            server_url = f"tcp://{self.config['SERVER_IP']}:{self.config['SERVER_PORT']}"
        else:
            server_url = f"ofi+tcp://{self.config['SERVER_IP']}:{self.config['SERVER_PORT']}"
        
        cmd = [
            'lightbeam_server_unit_test',
            '--transport', self.config['TRANSPORT_TYPE'],
            '--url', server_url
        ]
        
        if self.config['VERBOSE']:
            cmd.append('--verbose')
        
        Exec(' '.join(cmd), LocalExecInfo(env=self.env))

    def stop(self):
        try:
            Exec("pkill -f lightbeam_server_unit_test", LocalExecInfo(env=self.env))
        except:
            pass

    def clean(self):
        pass 