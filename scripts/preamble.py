"""
Prepends the preabmle to all files in the repo

USAGE:
    python3 scripts/preamble.py
"""

import sys,os,re
import pathlib

preamble = """/*
 * Copyright (C) 2022  SCS Lab <scslab@iit.edu>,
 * Luke Logan <llogan@hawk.iit.edu>,
 * Jaime Cernuda Garcia <jcernudagarcia@hawk.iit.edu>
 * Jay Lofstead <gflofst@sandia.gov>,
 * Anthony Kougkas <akougkas@iit.edu>,
 * Xian-He Sun <sun@iit.edu>
 *
 * This file is part of LabStor
 *
 * LabStor is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public
 * License along with this program.  If not, see
 * <http://www.gnu.org/licenses/>.
 */

"""

def PrependPreamble(path):
    text = ""
    with open(path, 'r') as fp:
        text = fp.read()
        if "Copyright (C) 2022  SCS Lab <scslab@iit.edu>" in text:
            text = text.replace(preamble, "")
        text = text.strip()
        text = re.sub("//\n// Created by [^\n]*\n//\n", "", text)
        text = preamble + text + "\n"

    with open(path, 'w') as fp:
        fp.write(text)


def LocateCppFiles(root):
    for entry in os.listdir(root):
        full_path = os.path.join(root, entry)
        if os.path.isdir(full_path):
            LocateCppFiles(full_path)
        elif os.path.isfile(full_path):
            file_ext = pathlib.Path(full_path).suffix
            if file_ext == '.cc' or file_ext == '.h':
                PrependPreamble(full_path)


root=pathlib.Path(__file__).parent.parent.resolve()
LocateCppFiles(root)