# Attention based Contextual Multi-View Graph Conlvolutional Networks (ACMV-GCNs)
グラフ構造のシーケンスデータを入力として受け取り，複数のグラフ隣接行列によって計算された出力値を注意機構によって融合し，予測を行うモデルです．  
layers配下にグラフのシーケンスデータに対する畳み込み演算を行うレイヤーが格納されています．  
レイヤーにおける演算などはhttps://github.com/vermaMachineLearning/keras-deep-graph-learning を参考にしており，
こちらのリンクのMultiGraphCNNを時系列データにおける回帰問題に対しても適用できるようにレイヤーの仕様を変更しました．

## 変更点
- シーケンスのデータを入力として受取り、シーケンスとして出力するように変更．
- メモリ使用量節約のため演算に使用する隣接行列の3次元テンソルを入力として与えるのではなく，予め演算に使用する隣接行列をレイヤーに与え，モデルの内部で演算に使用するように変更．

## モデル概要図
![architecture_poi](https://user-images.githubusercontent.com/56177314/115978686-31a0f980-a5bc-11eb-9d73-0f0bb6449a13.png)
