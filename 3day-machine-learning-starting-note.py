人工智能阶段
	机器学习 三天
	深度学习 三天
	量化交易 四天


传统的机器学习算法
	机器学习概述、特征工程 1天
	分类算法 1天
	回归算法、聚类 1天


机器学习概述
	1.1 人工智能概述
		达特茅斯会议 - 人工智能的起点
		机器学习是人工智能的一个实现途径
		深度学习是机器学习的一个方法发展而来
		1.1.2 机器学习、深度学习能做什么
			传统预测
			图像识别
			自然语言处理

	1.2 什么是机器学习
		数据
		模型
		预测
		从历史数据当中获得规律？这些历史数据是怎么样的格式？
		1.2.1 数据集构成
			特征集 + 目标值

	1.3 机器学习算法分类
		监督学习
			目标值：类别 - 分类问题
				KNN算法、贝叶斯分类、决策树与随机森林、逻辑回归
			目标值：连续性的数据 - 回归问题
				线性回归、岭回归
		目标值：无 - 无监督学习
			聚类 k-means
		1.预测明天的气温是多少度？ 回归
		2.预测明天是阴、晴还是雨？ 分类
		3.人脸年龄预测？ 回归/分类
		4.人脸识别？ 分类

	1.4 机器学习开发流程
		1）获取数据
		2）数据处理
		3）特征工程
		4）机器学习算法是训练 - 模型
		5）模型评估
		6）应用

	1.5 学习框架和资料介绍
		1）算法是核心、数据与计算是基础
		2）找准定位
		3）怎么做？
			1.入门
			2.实战类书籍
			3.机器学习 - '西瓜书' - 周志华
			  统计学习方法 - 李航
			  深度学习 - '花书'
		4）1.5.1 机器学习库与框架


特征工程
	2.1 数据集
		2.1.1 可用数据集
			公司内部 百度
			数据接口 收费
			数据集
			学习阶段可以用的数据集：
				1）sklearn
				2）kaggle
				3）UCI
			1 Scikit-learn工具介绍
		2.1.2 sklearn数据集
			sklearn.datasets
				load_* 获取小规模数据集
				fetch_* 获取大规模数据集
				2 sklearn小数据集
					sklearn.datasets.load_iris()
				3 sklearn大数据集
					sklearn.datasets.fetch_20newsgroups(data_home=None, subset='train')
				4 数据集的返回值
					datasets.base.Bunch(继承自字典)
						dict["key"] = values
						bunch.key = values
				思考：拿到的数据是否全部用来训练一个模型？
		2.1.3 数据集的划分
			训练数据：用于训练，构建模型
			测试数据：在模型检验时使用，用于评估模型是否有效
				测试集 20% ~ 30%
				sklearn.model_selection.train_test_split(arrays, *options)
				训练集特征值，测试集特征值，训练集目标值，测试集目标值
				x_train, y_train, x_test, y_test


	2.2 特征工程介绍
		算法 特征工程
		2.2.1 为什么需要特征工程(Feature Engineering)
		2.2.2 什么是特征工程
			sklearn 特征工程
			pandas 数据清洗、数据处理
				特征抽取/特征提取(Feature Extraction)
					机器学习算法 - 统计方法 - 数学公式
						文本类型 -> 数值
						类型 -> 数值
					2.3.1 特征提取
						sklearn.feature_extraction
					2.3.2 字典特征抽取 - 类别 -> one-hot编码
						sklearn.feature_extraction.DictVectorizer(sparse=True,...)
						vector 数学：向量 物理：矢量
							矩阵 matrix 二维数组
							向量 vector 一维数组
						父类：转换器类
						返回sparse矩阵
							sparse稀疏
								将非零值 按位置表示出来
								节省内存 - 提高加载效率
						应用场景：
							1）pclass, sex 数据集中类别特征比较多
								1、将数据集的特征 -> 字典类型
								2、DictVectorizer转换
							2）本身拿到的数据就是字典类型
					2.3.3 文本特征提取：对文本数据进行特征值化
						单词作为特征
						句子、短语、单词、字母
						特征：特征词
						方法1：CountVectorizer 统计每个特征词出现的次数
							sklearn.feature_extraction.text.CountVectorizer(stop_words=[]) 返回词频矩阵
								stop_words 停用词
								停用词表
							关键词：在某一个类别的文章中，出现的次数很多，但是在其他类别的文章当中出现很少
							CountVectorizer.fit_transform(X) X:文本或者包含文本字符串的可迭代对象 返回sparse矩阵
							CountVectorizer.inverse_transform(X) X:array数组或者sparse矩阵 返回转换之前的数据格
							CountVectorizer.get_feature_name() 返回单词列表
						方法2：TfidfVectorizer
							sklearn.feature_extraction.text.TfidVectorizer
							Tf-IDF作用：评估重要程度
								eg: 两个词 "经济" "非常"
									1000篇文章 - 语料库
									100篇文章 - "非常"
									10篇文章 - "经济"
									两篇文章 -> 
											文章A(100词)：10次"经济" TF-IDF: tf * idf = 0.2
												tf：10/100=0.1
												idf：log 10 (1000/10) = 2
								        	文章B(100词): 10次"非常" TF-IDF: tf * idf = 0.1
								        		tf：10/100=0.1
								        		idf：log 10 (1000/100) = 1
							TF - 词频(term frequency)
							IDF - 逆文档频率(inverse document frequency)：总文件数目 / 包含该词语的文件的数目，再将得到的商取以10为底的对数


				特征预处理(Feature Preprocessing)
					2.4.1 什么是特征预处理
						为什么我们要进行归一化/标准化？
							无量纲化
						2.4.2 归一化
							sklear.preprocessing.MinMaxScaler
							通过对原始数据进行变换把数据映射到(默认为[0，1])之间
							异常值：最大值、最小值
							这种方法鲁棒性较差，知识和传统精确小数据场景
						2.4.3 标准化
							sklearn.preprocessing.StandardScaler
							通过对原始数据进行变换把数据变换到均值为0，标准差为1的范围内
							(x - mean) / std
							标准差：集中程度
							应用场景：
								在已有样本足够多的情况下比较稳定，适合现代嘈杂大数据场景。

				特征降维(Feature Dimension Reduction)
					2.5.1 降维 - 降低维度
						ndarray
							维数：嵌套的层数
							0维：标量
							1维：向量
							2维：矩阵
							3维
							n维
						二维数据(处理对象)
							此处的降维：
								降低特征的个数
							效果：
								特征与特征之间不相关
						特征选择(sklearn.feature_selection)
							Filter过滤器
								方差选择法：低方差特征过滤
								相关系数 - 特征与特征之间的相关程度
									取值范围： -1 <= r <= +1
									皮尔逊相关系数(Pearson Correlation Coefficient)
									scipy.stats.person
									特征与特征之间相关性很高：
										1）选取其中一个
										2）加权求和
										3）主成分分析
							Embeded嵌入式
								决策树 day2
								正则化 day3
								深度学习 day5
						主成分分析
							2.6.1 什么是主成分分析(PCA)
								sklearn.decomposition.PCA(n_components=None)
								n_components:
									小数：表示保留百分之多少的信息
									整数：减少到多少特征
							2.6.2 案例：探究用户对物品类别的喜好细分
								用户 			物品类别
								user_id 		aisle
								1）需要将user_id和aisle放在同一个表中
								2）找到user_id和aisle - 交叉表和透视表
								3）特征冗余过多 -> PCA降维


分类算法
目标值：类别
1、sklearn转换器和预估器
2、KNN算法
3、模型选择与调优
4、朴素贝叶斯算法
5、决策树
6、随机森林

	3.1 sklearn转换器和预估器
		转换器
		预估器(estimator)
		3.1.1 转换器 - 特征工程的父类
			1 实例化(实例化的是一个转换器类(Transform))
			2 调用fit_transform(对于文档建立分类词频矩阵,不能同时调用)
			标准化：
				(x - mean) / std
				fit_transform()
					fit()			计算每一列的平均值、标准差
					transform()		(x - mean) / std 进行最终的转换
		3.1.2 预估器(sklearn机器学习算法的实现)
			预估器(estimator)
				1 实例化一个estimator
				2 estimator.fit(x_train, y_train) 计算
					- 调用完毕,模型生成
				3 模型评估:
					1）直接比对真实值和预测值
						y_predict = estimator.predict(x_test)
						y_test == y_predict
					2）计算准确率
						accuracy = estimator.score(x_test, y_test)

	3.2 KNN算法
		3.2.1 什么是KNN算法
			KNN核心思想：
				你的"邻居"来推出你的类别
			1 KNN算法原理
				k = 1
					容易受到异常点的影响
				如何确定谁是邻居？
				计算距离：
					距离公式
						欧氏距离
						曼哈顿距离 绝对值距离
						明可夫斯基距离
			2 电影类型分类案例
				如果取的最近的电影数量不一样，会是什么结果？
					K 值取的过小 容易受到异常点的影响
					K 值取的过大 容易收到样本不均衡的影响
				结合前面的约会对象数据，分析KNN算法需要什么样的处理
					无量纲化的处理
						标准化
				sklearn.neighbotd.KNeighborsClassifier(n_neighbors=5, algorithm='auto')
					n_neighbors: K值
		3.2.3 案例1：鸢尾花种类预测
			1）获取数据
			2）数据集划分
			3）特征工程
				标准化
			4）KNN预估器流程
			5）模型评估
		3.2.4 KNN总结
			优点：简单，易于理解，易于实现，无需训练
			缺点：
				1）必须指定K值，K值选择不当则分类精度不能保证
				2）懒惰算法，对测试样本分类时的计算量大，内存开销大
			使用场景：小数据场景，几千至几万样本，具体场景具体业务去测试
					

	3.3 模型选择与调优
		3.3.1 什么是交叉验证(Cross Validation)
			交叉验证：将拿到的训练数据，分为训练集和验证集。
				eg:将数据分成四份，其中一份作为验证机，经过4次测试，每次更换不同的验证集，得到四组模型结果，取平均值作为最终结果，又称4折交叉验证
			目的：为了让被评估的模型更加准确可信
		3.3.2 超参数搜索-网格搜索(Grid Search)
			K的取值
				[1, 3, 5, 7, 9]
				暴力破解
		3.3.3 鸢尾花案例增加K值调优
		3.3.4 案例：预测Facebook签到位置
			kaggle competition
			流程分析：
				1）获取数据
				2）数据处理
					目的：
						特征值 x
						目标值 y
						a.缩小数据范围(根据坐标选区范围)
							2 < x < 2.5
							1 < y < 1.5
						b.time -> 年月日时分秒
						c.过滤签到次数少的地点
						数据集划分
				3）特征工程：标准化
				4）KNN算法预估流程
				5）模型选择与调优
				6）模型评估
	
	3.4 朴素贝叶斯算法
		sklearn.naive_bayes.MultinomialNB()
		3.4.1 什么是朴素贝叶斯算法
			朴素贝叶斯法是基于贝叶斯定理与特征条件独立假设的分类方法。
			最为广泛的两种分类模型是决策树模型(Decision Tree Model)和朴素贝叶斯模型（Naive Bayesian Model，NBM）
		3.4.2 联合概率、条件概率与相互独立
			- 联合概率：
				包含多个条件, 且所有条件同时成立的概率
				P(A, B)
				eg: P(程序员, 匀称)
					P(程序员, 超重|喜欢)
			- 条件概率：
				事件A在另一事件B已经发生条件下的发生概率
				P(A|B)
				eg: P(程序员|喜欢)
					P(程序员, 超重|喜欢)
			- 相互独立：
				P(A, B) = P(A)P(B) <==> 事件A与事件B相互独立
		3.4.3 贝叶斯公式
			朴素？
				假定了特征与特征相互独立
			朴素贝叶斯算法：
				朴素 + 贝叶斯
			应用场景：
				文本分类
				单词作为特征
			拉普拉斯平滑系数
		3.4.5 案例：20类新闻分析
			1）获取数据
			2）划分数据集
			3）特征工程
				文本特征抽取
			4）朴素贝叶斯预估器流程
			5）模型评估
		3.4.6 朴素贝叶斯算法总结
			优点：
				发源于古典数学理论, 有稳定的分类效率
				对缺失数据不敏感, 算法简单, 常用于文本分类
				准确度高, 速度快
			缺点：
				由于假定特征与特征之间相互独立, 如果特征属性有关联时效果不好

	3.5 决策树
		sklearn.tree.DecisionTreeClassifier(criterion='gini', max_depth=None, random_state=None)
		决策树思想来源朴素, 程序设计中的条件分支结构就是if-else结构, 最早的决策树就使用这类结构分割数据
		3.5.1 认识决策树
			如何高效地进行决策？
				特征的先后顺序
		3.5.2 决策树分类原理详解
			已知 四个特征 预测 是否贷款给某个人
			先看房子, 在工作 -> 是否贷款 只看两个特征
			先看年龄, 信贷情况, 工作 -> 三个特征
		信息论基础
			1）信息
				香农：是消除随机不定性的东西
				小明 年龄 '我今年18岁' - 信息
				小华 '小明明年19岁' - 已经在上一句消除了, 不是信息
			2）信息的衡量 - 信息量 - 信息熵
				bit
				g(D,A) = H(D) - 条件熵H(D|A)
			3）决策树的划分依据之一 -- 信息增益
		3.5.3 决策树可视化
			sklearn.tree.export_graphviz(estimator, out_file='tree.dot', feature_name=['','']) -> 导出DOT格式
		3.5.4 决策树总结
			优点:
				可视化 - 可解释能力强
			缺点:
				容易产生过拟合
		3.5.5 案例:泰坦尼克号乘客生存预测
			流程分析：
				特征值 目标值
				1）获取数据
				2）数据处理
					缺失值处理
					特征值 -> 字典类型
				3）准备好特征值 目标值
				4）划分数据集
				5）特征工程：字典特征值抽取
				6）决策树预估器流程
				7）模型评估

	3.6 集成学习方法之随机森林
		3.6.1 什么是集成学习方法
			集成学习通过建立几个模型组合来解决单一预测问题, 工作原理是生成多个分类器/模型, 各自独立学习和预测。
			这些预测最后结合成组合预测, 因此由于任何一个单分类的做出预测。
		3.6.2 什么是随机森林
			随机
			森林：包含多个决策树的分类器
		3.6.3 随机森林原理过程
			训练集：
				N个样本
				特征值 目标值
				M个特征
			随机
				两个随机
					训练集随机 -> N个样本中随机有放回抽样N个
						bootstrap - 随机有放回抽样
						[1, 2, 3, 4, 5]
						新的树的训练集
						[2, 2, 3, 1, 5]
					特征随机 -> M个特征中随机抽取m个特征
						M >> m
						降维
		3.6.4 API
			sklearn.ensemble.RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, bootstrap=True,
													random_state=None, min_samples_split=2)
							n_estimators:森林里的数目数量 default=10
							criterion:分割特征的测量方法 default='gini'
							max_depth:树的最大深度 defaul=None
							max_features='auto':每个决策树的最大特征数量
									if auto: max_features=sqrt(n_features)
									if sqrt: same as 'auto'
									if log2: max_features=log2(n_features)
									if None max_features=n_features
							bootstrap: 是否在构建树时使用放回抽样 default=True
							min_samples_split 节点划分最少样本数
							min_samples_leaf: 叶子节点的最小样本数
		3.6.5 总结
			- 极好的准确率
			- 能有效运行在大数据集上, 处理具有高维特征的输入样本, 而且不需要降维
			- 能评估各个特征在分类问题上的重要性

回归和聚类
	线性回归
	欠拟合、过拟合
	岭回归

分类算法：逻辑回归
模型保存与加载
无监督学习：K-means

	4.1 线性回归
		回归问题：
			目标值 - 连续型的数据
		4.1.1 线性回归的原理
			什么是线性回归？
				函数关系 特征值和目标值
				- 线性模型： y = w1x1 + w2x2 +w3x3 + ...... + wnxn +b
				  		    = wTx + b
				数据挖掘基础
				y = kx + b
				y = 0.7x1 + 0.3x2
				期末成绩： 0.7 x 考试成绩 + 0.3 x 平时成绩
				[[90, 85],
				[]]
				[[0.3],
				[0.7]]
				[8, 2] * [2, 1] = [8, 1]
				- 广义线性模型
					非线性关系？
					线性模型：
						自变量一次 -> y = w1x1 + w2x2 +w3x3 + ...... + wnxn +b
						参数一次 -> y = w1x1 + w2x1^2 + w3x1^3 + w4x1^4 + ...... + b
				- 线性关系 & 线性模型：
					1 线性关系一定是线性模型
					2 线性模型不一定是线性关系
		4.1.2 线性回归的损失和优化原理(理解记忆)
			目标：求模型参数
				模型参数能够使得预测准确
			真实关系：真实房子价格 = 0.02x市中心距离 + 0.04x城市一氧化氮浓度 + (-0.12x自住房平均房价) + 0.25x城镇犯罪率
			随意假定：预测房子价格 = 0.25x市中心距离 + 0.14x城市一氧化氮浓度 + 0.42x自住房平均房价 + 0.34x城镇犯罪率
			损失函数/cost/成本函数/目标函数：
				最小二乘法
			优化损失
				优化方法？
				正规方程
				sklearn.linear_model.LinearRegression(fit_intercept=True)
							fit_intercept:是否计算偏置
							coef_：回归系数
							intercept_：偏置
					天才 - 直接求解W
					拓展：
					1）
						y = ax^2 + bx +c
						y‘ = 2ax + b = 0
						x = -b / 2a
					2）
						a * b = 1 
							b = 1 / a = a ^ - 1
						A * B = E
						[[1, 0, 0],
						[0, 1, 0],
						[0, 0, 1]]
						B = A ^ -1
				梯度下降
				sklearn.linear_model.SGDRegression(loos='squared_loss', fit_intercept=True, learning_rate='invscaling', eta0=0.01)
							loss:损失类型 -> 'squared_loss'普通最小二乘法
							learning_rate:学习率填充
							coef_：回归系数
							intercept_：偏置
					勤奋努力的普通人
						试错、改进
		4.1.3 波士顿房价预测
			流程分析：
				1）获取数据
				2）划分数据集
				3）特征工程
					无量纲化 - 标准化
				4）预估器流程
					fit() -> 模型
					coef_ intercept_
				5）模型评估
		4.1.4 回归的性能评估
			均方误差(Mean Squared Error -> MSE)评价机制：
				sklearn.metrics.mean_squared_error(y_true, y_pred)
					均方误差回归损失
					y_true 真实值
					y_pred 预测值
					return 浮点数结果
			正规方程和梯度下降对比：
						梯度下降					正规方程
					需要选择学习率			不需要选择学习率
					需要迭代求解				一次运算
					特征数量较大可以使用		需要计算方程,时间复杂度高O(n3)
		4.1.5 拓展 - 关于优化方法GD\SGD\SAG
			1 GD
				梯度下降(Gradient Descent), 原始梯度下降法需要计算所有样本的值才能得出梯度, 计算量大。
			2 SGD
				随机梯度下降(Stochastic Gradient Descent)是一个优化方法,在一次迭代时只考虑一个训练样本。
				优点：
					高效
					容易实现
				缺点：
					需要许多超参数 -> 比如正则项参数, 迭代数
					对于特征标准化敏感
			3 SAG
				随机平均梯度法(Stochastic Average Gradient), 由于收敛速度太慢, 有人提出这种基于梯度下降的算法。 
				岭回归和逻辑回归都会有SAG优化

	4.2 欠拟合与过拟合
		训练集表现的好, 测试集表现不好 --> 过拟合
		4.2.1 什么是欠拟合与过拟合
			欠拟合
				学习到的特征过少, 模型过于简单
				解决:
					增加数据的特征数量
			过拟合
				学习到的特征过多, 模型过于复杂
				原始特征过多, 存在一些嘈杂特征, 模型过于复杂是因为模型尝试去兼顾各个测试数据点
				解决:
					正则化
						L1 	损失函数 + λ惩罚项 
							--LASSO
						L2->更常用
							作用：使其中一些W都很小, 接近于0, 削弱某个特征的影响
							损失函数 + λ惩罚项
							--Ridge(岭回归)

	4.3 线性回归的改进 - 岭回归
		4.3.1 带有L2正则化的线性回归 - 岭回归
			sklearn.linear_model.Ridge(alpha=1.0, fit_intercept=True, solver='auto', normalize=False)
				alpha:正则化力度=惩罚项系数λ
				solver:根据数据自动选择优化方法
				normalize:数据是否进行标准化(False则在fit之前标准化, True则不进行标准化操作)
	
	4.4 分类算法 - 逻辑回归与二分类
		4.1.1 逻辑回归的应用场景
			- 广告点击率
			- 是否为垃圾邮件
			- 是否患病
			- 金融诈骗
			- 虚假账号
			都是两个类别之间的判断, 解决二分类问题的利器
		4.4.2 逻辑回归的原理
			1.输入
				逻辑回归的输入 = 线性回归的输出
			2.激活函数
				sigmoid函数
				1 / (1 + e^(-x))
				回归的结果输入到sigmoid函数中
				输出结果：[0, 1]区间中的一个概率值, 默认0.5为阈值 -> 大于阈值即属于该类型, 小于阈值即不属于
			3.假设函数/线性模型
				1 / (1 + e^(-(w1x1 + w2x2 + w3x3 + ...... + wnxn + b)))
			4.损失函数
				(y_predict - y_true)平方和 / 总数
				逻辑回归的真实值 是否属于某个类别
				逻辑回归的损失 - '对数似然损失'
				log 2 x
			5.优化损失
				梯度下降
		4.4.3 逻辑回归API
			sklearn.linear_model.LogisticRegression(solver='libinear', penalty='l2', C=1.0)
					solver:优化求解方式(默认开源的liblinear库实现, 内部使用了坐标轴下降法来迭代优化损失函数)
							solver=sag 根据数据集自动选择, 随机平均梯度下降
					penalty:正则化的种类
					C:正则化力度
		4.4.4 案例:癌症分类预测 - 良/恶性乳腺癌肿瘤预测
			流程分析：
				1）获取数据
					读取的时候加上names
				2）数据处理
					处理缺失值
				3）数据集划分
				4）特征工程：
					无量纲化处理 - 标准化
				5）逻辑回归预估器
				6）模型评估
		4.4.5 分类的评估方法
		sklearn.metrics.classification_report(y_true, y_pred, labels=[], target_names=None)
						y_true:真实目标值
						y_pred:预估器预测目标值
						labels:指定类别对应的数字
						target_names:目标类别名称
			1.精确率与召回率
				- 混淆矩阵
							   正例	 	 假例
						正例	  真正例		伪反例
						假例	  伪正例		真反例
					精确率 -> 预测结果为正例样本中真实为正例的比例
					召回率 -> 真实为正例的样本中预测结果为正例的比例
					TP = True Possitive
					FN = False Negative
				- 精确率(Precision)与召回率(Recall)
					精确率
					召回率 查的全不全
				- F1-score 模型的稳健性
			e.g.总共有100个人, 99个样本癌症, 1个样本非癌症
				无论如何全都预测正例, 默认癌症为正例
				准确率:99%
				召回率:99 / 99 = 100%
				精确率:99%
				F1-score:(2 * 99% * 100%) / 199% = 99.497%
				AUC:0.5 -> 最差的, 说明模型不好
					TPR = 100%
					FPR = 1/1 = 100%
			2.ROC曲线与AUC指标
				- 知道TPR与FPR
					TPR = TP / (TP+FN)
						所有真实类别为1的样本中, 预测类别为1的比例
					FPR = FP / (FP+TN)
						所有真实类别为0的样本中, 预测类别为1的比例
				- ROC曲线
					横轴是FPR
					纵轴是TPR
					二者相等时代表不论真实类别为1还是0, 分类器预测为1的概率相等, 此时AUC为0.5
				- AUC指标
					概率意义是随机取一对正负样本, 正样本得分大于负样本的概率
					[0.5, 1] 取值越高越好
					AUC=1 -> 完美分类器
					0.5 < AUC < 1 优于随机猜测
				- AUC计算API
					sklearn.metrics.roc_auc_score(y_true, y_score)
							计算ROC曲线面积, 即AUC值
							y_true 每个样本的真实类别, 必须为0 or 1标记
							y_score 预测得分, 可以是正类估计概率、置信值或分类器方法的返回值
				- 总结
					只能用来评价二分类
					非常适合评价样本不平衡中的分类器性能
		4.4.6 sklearn模型的保存和加载
			from sklearn.externals import joblib
			# import joblib
				保存: joblib.dump(rf, 'test.pkl')
				加载: estimator = joblib.load('test.pkl')

	4.5 无监督学习和K-means算法
		4.5.1 什么是无监督学习
			没有目标值 - 无监督学习
		4.5.2 无监督学习包含算法
			聚类
			K-means(K均值聚类)
			降维
			PCA
		4.5.3 K-means API
			sklearn.cluster.KMeans(n_clusters=8, int='k-means++')
					k-means:聚类
					n_clusters:开始的聚类中心数量
					init:初始化方法, 默认为'k-means++'
					labels_:默认标记的类型, 可以和真实值比较(不是值比较)
		4.5.4 案例:K-means对Instacart Market用户聚类
			k = 3
			流程分析：
			降维之后的数据
			1）预估器流程
			2）看结果
			3）模型评估
		4.5.5 KMeans性能评估指标
			- 轮廓系数
				如果	b_i >> a_i: 趋近于1效果更好
					b_i << a_i: 趋近于-1效果不好
				轮廓系数的值介于[-1, 1], 越趋近于1代表内聚度和分离度都相对较优
			- 轮廓系数API
				sklearn.metrics.silhouette_score(X.labels)
					计算所有样本的平均轮廓系数
					X：特征值
					labels：被聚类标记的目标值
		4.5.6 KMeans总结
			优点
				采用迭代式算法, 直观易懂并且非常实用
			缺点
				容易收敛到局部最优解(多次聚类)
			应用场景
				没有目标值
				分类





				
