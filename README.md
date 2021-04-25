# Attention based Contextual Multi-View Graph Conlvolutional Networks (ACMV-GCNs)
グラフ構造のシーケンスデータを入力として受け取り，複数のグラフ隣接行列によって計算された出力値を注意機構によって融合し，予測を行うモデルです．  
layers配下にグラフのシーケンスデータに対する畳み込み演算を行うレイヤーが格納されています．  
レイヤーにおける演算などはhttps://github.com/vermaMachineLearning/keras-deep-graph-learning を参考にしており，
こちらのリンクのMultiGraphCNNを時系列データにおける回帰問題に対しても適用できるようにレイヤーの仕様を変更しました．

## 変更点
主な変更点は，
- シーケンスのデータを入力として受取り、シーケンスとして出力するように変更．
- メモリ使用量節約のため演算に使用する隣接行列の3次元テンソルを入力として与えるのではなく，予め演算に使用する隣接行列をレイヤーに与え，モデルの内部で演算に使用するように変更．
の2点が挙げられます．

従って，レイヤーへの入力は [batch_size, num_graph_nodes, input_dim] から [batch_size, sequence_length, num_graph_nodes, graph_conv_filters] の四次元テンソルに変更されています．

## データ
データは株式会社ドコモ・インサイトマーケティングよりご提供頂いた「モバイル空間統計」を使用しています．
