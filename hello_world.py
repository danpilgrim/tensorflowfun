

import tensorflow as ai

hello = ai.constant('hello world !!')

sess = ai.Session()

print(sess.run(hello))
