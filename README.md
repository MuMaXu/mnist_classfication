# mnist_classfication
mnist_classfication using Tensorflow
关于代码中几点重要的说明，也当是一个知识的复习回顾

一、经典的损失函数
1.交叉熵损失函数
H(p,q)=−∑8_x▒〖p(x)  log⁡q(x) 〗，p是真实的分布，q是模型给出的分布
刻画了两个概率分布之间的距离,一般需要加一个softmax层将输出层变为概率分布，交叉熵越小，越说明两个分布之间的距离就越小。
代码：tf.reduce.mean(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))
2.均方误差
MSE(y,y^′ )=(∑24_(i=1)^n▒〖(y_i−y_i^ )〗)/n，常用于回归问题
代码：mse=tf.reduce_mean(tf.square(y_-y))
3.交叉熵和softmax回归
代码：cross_entrop=tf.nn.softmax_cross_entrop_with_logits(labels=y_,logits=y)
	    loss=tf.reduce_mean(cross_entrop)
说明：labels表示训练数据的正确答案，只需要传入正确答案的数字就可以，常用tf.argmax(y_,1)
          logits表示训练的结果
作用：在只有一个正确答案的分类问题中，使用这个函数可以进一步加速计算过程，因为相当于只有一个数来乘以一个向量了，而不是两个向量相乘

二、神经网络训练过程
batch_size=n
x=tf.placeholder(tf.float32.shape=(batch_size,2),name="x-input")
y_=tf.placeholder(tf.float32.shape=(batch_size,1),name="y-input")

loss=…
train_op=tf.train.AdamOptimizer(0.001).minimize(loss)

with tf.Session() as sess:
	init_op=tf.global_variables_initializer()
	sess.run(init_op)
	
	for I in range(STEPS):
		current_x,current_y=…
		sess.run(train_op,feed_dict={x:current_x,y_:current_y})
		
三、学习率的设置
目的是让学习率在刚开始的时候比较大，然后慢慢减小到接近局部最优解
代码实现：
①global_step=tf.Variable(o)   #在minimize函数中传入global_step将自动更新
②tf.train.exponential_decay(0.1,global,100,0.96,staircase=True)
③learning_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(global_step=global_step)
代码解释：设置初始学习率为0.1，因为设定了staircase=True，所以训练每经过100次训练学习率就会乘以0.96
代码原形：decayed_learning_rate=learning_rate*decay_rate^(global_step/decay_steps)

四、过拟合问题
代码实现：tf.contrib.layers.l2_regularizer(lambda)(w)

五、滑动平均模型
用途：用于控制变量的更新幅度，使得模型在训练初期参数更新较快，在接近最优值处参数更新较慢，幅度较小
方式：主要通过不断更新衰减率来控制变量的更新幅度
数学公式：shadow_variable=decay*shadow_variable+(1-decay)*variable
解释：shadow_variable为影子变量，decay为衰减率，variable为待更新的变量
代码实现：
①定义一个滑动平均的类，初始化时给了衰减率（0.99），和控制衰减率的变量
ema=tf.train.ExponentialMovingAverage(0.99,step)
②定义一个滑动平均的操作，这里需要给出一个列表，每次执行这个操作的时候这个列表中的变量都会被更新
maintain_average_op=ema.apply([v])
③获取滑动平均值
sess.run(maintain_average_op)
sess.run(ema.average([v]))



			


六、变量的管理
代码一：tf.get_variable(name="v",shape=[],initializer=tf.constant_initializer(1.0))
作用：创建或者获取一个变量
与tf.Variables()的区别：在tf.Variables()中name="v"不是必须的，但是在tf.get_variable()中是必须的
注意：如果有同名的变量，也就是说有另外的一个变量的名字也是“v”，那么代码一就会报错
代码二：tf.variable_scope("foo")
作用：生成一个上下文管理器，如果添加参数reuse=True的话，那么可以tf.get_variable函数就会直接获取已经创建的变量，如果这个变量没有的话就会报错；如果添加参数reuse=None或者reuse=False的话，tf.get_variable函数将创建新的变量，如果这个变量存在的话就会报错
代码三：v.name
作用：获取变量的命名空间，也即是foo/v:0

七、tensorflow模型持久化
作用一：如果我们的神经网络比较复杂，训练数据比较多，那么我们的模型训练就会耗时很长，如果在训练过程中出现某些不可预计的错误，导致我们的训练意外终止，那么我们将会前功尽弃。为了避免这个问题，我们就可以通过模型持久化（保存为CKPT格式）来暂存我们训练过程中的临时数据。
作用二：如果我们训练的模型需要提供给用户做离线的预测，那么我们只需要前向传播的过程，只需得到预测值就可以了，这个时候我们就可以通过模型持久化（保存为PB格式）只保存前向传播中需要的变量并将变量的值固定下来，这个时候只需用户提供一个输入，我们就可以通过模型得到一个输出给用户。
保存为 CKPT 格式的模型
流程：
①.定义运算过程
②.声明并得到一个 Saver
③.通过 Saver.save 保存模型
代码实现：
MODEL_DIR = "model/ckpt" 
MODEL_NAME = "model.ckpt"
"""
定义计算部分
"""
saver = tf.train.Saver()    #声明一个Saver()类saver用于保存模型
with tf.Session() as sess:
	saver.save(sess, os.path.join(MODEL_DIR, MODEL_NAME))    #模型保存

代码生成结果：
	• checkpoint ： 记录目录下所有模型文件列表
	• ckpt.data ： 保存模型中每个变量的取值
	• ckpt.meta ： 保存整个计算图的结构

保存为 PB 格式模型
流程：
	①. 定义运算过程
	②. 通过 get_default_graph().as_graph_def() 得到当前图的计算节点信息
	③. 通过 graph_util.convert_variables_to_constants 将相关节点的values固定
	④. 通过 tf.gfile.GFile 进行模型持久化

代码实现：
# MODEL_DIR = "model/pb" 
# MODEL_NAME = "addmodel.pb"
"""
定义计算部分
"""
with tf.Session() as sess:
	graph_def = tf.get_default_graph().as_graph_def() #得到当前的图的 GraphDef 部分，通过这个部分就可以完成重输入层到输出层的计算过程
	output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
	        sess,
	        graph_def,
	        ["predictions"] #需要保存节点的名字
	    )
	with tf.gfile.GFile(output_graph, "wb") as f: # 保存模型                     
		f.write(output_graph_def.SerializeToString()) # 序列化输出
	
	
*GraphDef：这个属性记录了tensorflow计算图上节点的信息。
代码生成结果：
	• add_model.pb ： 里面保存了重输入层到输出层这个计算过程的计算图和相关变量的值，我们得到这个模型后传入一个输入，既可以得到一个预估的输出值
