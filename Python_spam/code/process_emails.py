#!/usr/bin/env python3
# process_emails.py - Process emails from SpamAssassin dataset

import os
import email
from collections import Counter

targets = []
msgs = []
txt_msg = []
types = Counter()

for root, dirs, files in os.walk("../data/"):
    for file in files:
        abs_path = os.path.join(root, file)
        try:
            if file.endswith("ipynb"):
                pass
            elif "ham" in abs_path:
                with open(abs_path, "r", encoding='latin-1') as f:
                    msg = email.message_from_file(f)
                    txt = msg.get_payload()
                    type_ = msg.get_content_type()
                txt_msg.append(txt)
                targets.append(0)
                msgs.append(file)
                types[type_]+=1
            elif "spam" in abs_path:
                with open(abs_path, "r", encoding='latin-1') as f:
                    msg = email.message_from_file(f)
                    txt = msg.get_payload()
                    type_ = msg.get_content_type()
                txt_msg.append(txt)
                targets.append(1)
                msgs.append(file)
                types[type_]+=1
            else:
                print(f"something happened you didn't expect for file {abs_path}")
        except Exception as e:
            print(f"Error processing {abs_path}: {str(e)}")

# Print summary of processed files
print(f"Total emails processed: {len(targets)}")
print(f"Ham emails: {targets.count(0)}")
print(f"Spam emails: {targets.count(1)}")
print(f"Content types: {dict(types)}") 