# Copyright (c) 2024
#
# Authors:
# Sen2Cor_dev_team (Telespazio)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module to manage user credentials"""

import getpass
import logging
import os
import pathlib
import sys

import rsa
import yaml

log = logging.getLogger("pythonConfig")


class Credentials:
    def __init__(self):
        self.public_key = None
        self.private_key = None
        self.id = "2399898002@qq.com"
        self.password = "CZ123456789cz!!"

        if os.name == "nt":
            home_dir = os.getenv("USERPROFILE")
        else:
            home_dir = os.getenv("HOME")

        private_file_path = os.path.join(home_dir, ".private_key_dem_downloader.rsa")
        public_file_path = os.path.join(home_dir, ".public_key_dem_downloader.rsa")
        if os.path.exists(private_file_path) and os.path.exists(public_file_path):
            # Retrieval of public and private keys
            log.info("Retrieval of public and private keys")
            with open(private_file_path, "rb") as private_file:
                self.private_key = rsa.PrivateKey.load_pkcs1(private_file.read())
            with open(public_file_path, "rb") as public_file:
                self.public_key = rsa.PublicKey.load_pkcs1(public_file.read())
        else:
            # Generating public and private keys
            log.info("Generating public and private keys")
            (self.public_key, self.private_key) = rsa.newkeys(1012)
            with open(private_file_path, "w") as private_file:
                private_file.write(self.private_key.save_pkcs1().decode("utf8"))
            with open(public_file_path, "w") as public_file:
                public_file.write(self.public_key.save_pkcs1().decode("utf8"))

        file = open(
            pathlib.Path(os.path.realpath(__file__)).parent / "credentials.yaml",
            "r+",
            encoding="UTF-8",
        )
        auth = yaml.full_load(file) or {}
        username = getpass.getuser()
        if username in list(auth.keys()):
            log.info("User is known and id and password are retrieved")
            self.id, cipherpassword = auth[username]
            self.password = rsa.decrypt(cipherpassword, self.private_key)
        else:
            log.info(f"username: {self.id}, password: {self.password}")
            # log.info("Please enter your username and password")

            # self.id = input("Enter your username: ")
            while "@" not in self.id:
                log.warning("Your username shall be a valid email address")
                self.id = input("Enter your username: ")

            # self.password = getpass.getpass("Enter your password: ")
            password_check = self.password
            # password_check = getpass.getpass("Re-enter your password: ")
            if password_check == self.password:
                cipherpassword = rsa.encrypt(str.encode(self.password), self.public_key)
                auth[getpass.getuser()] = (self.id, cipherpassword)
                yaml.dump(auth, file)
                log.info(
                    "Password has been encrypted and stored in credentials.yaml for username %s",
                    self.id,
                )
            else:
                log.error("Passwords do not match. Please try to reset your credentials again")
                log.warning("Process will stop")
                sys.exit(1)

        file.close()
