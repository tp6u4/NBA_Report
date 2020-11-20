# NBA_Report
Integrate PCA, kNN, K-Means Algorithm 
---
title: "NBA.Position_Clustering_Report"
author: "Max Chen"
date: "2020/11/8"
output: 
    html_document:
        fig_width: 7 #圖片寬度
        fig_height: 5 #圖片高度
        theme: cerulean #主題
        highlight: tango
        toc: true     #設定標題深度 1~6
        toc_depth: 3  #標題階層深度(defalut是3)
        toc_float:
            collapsed: TRUE   #是否僅顯示top-level 
            smooth_scroll: False #是否隨著滾輪進行動畫
---

```{r setup, include=FALSE}
require(knitr)
require(kableExtra)
require(reshape2)
require(ggplot2)
require(plotly)
require(tidyverse)
require(dplyr)
require(nsprcomp)
require(cluster)    
require(factoextra) #找最佳k值
require(gridExtra)
require(readxl)
require(magrittr)
require(plotly)
require(useful)
library(GGally)
library(class) #knn
require(fastcluster) #加速版的hclust(階層式分群)
library(gridExtra) #ggplot排版(一次顯示多圖)
knitr::opts_chunk$set(echo = FALSE, warning = FALSE) 
```

# 零.NBA報告-資料分析及建模處理步驟
> **使用NBA 2018-19球季 球員平均數據作分群  R Version Built with`r getRversion()`**  
    
    壹.資料預處理及分析  
    貳.Principal Component Analysis篩選(變數降維度)  
    參.kNN分類  
    肆.Hierarchical Cluster分群  
    伍.K-Means分群
    陸.K-Medoid分群
    柒.決定最適分群數目
    
# 壹.資料預處理及分析  

**Problem：透過籃球基礎數據，分類球員位置並且檢視分類正確率**  
**Data：NBA 2018-19賽季資料**  
**目標變數(1)：Pos**  
**最終所使用變數(18)："MP" "FG" "FG." "X3P" "X3P." "X2P" "X2P." "eFG." "FT" "FT." "ORB" "DRB" "AST" "STL" "BLK" "TOV" "PF" "PTS" ** 

```{r 匯入NBA2018-19賽季資料作kNN}
nbaplayer_201819 <- read.csv("/Users/sky/Documents/統計模型實作/資料分群Cluster//NBA Players Stats 201819.csv")
kable(nbaplayer_201819[1:6, ]) %>% kable_styling() %>% row_spec(0, background = "red", color = "white") %>% scroll_box(height = "200px")
dim(nbaplayer_201819)
```

**由於有8位預定Pos只會有C,PF,SF,SG,PG，故將重複位置之球員修改為一個位置(詢問較為熟悉球員打法朋友)，並且將變數Pos轉變為factor類型變數**  
```{r}
nbaplayer_201819[grep("-", nbaplayer_201819$Pos), c("Player", "Pos")]
```


**分類各位置球員數如表及bar chart，檢視目前有79筆資料有NA值，後續處理**
```{r, echo = TRUE}
nbaplayer_201819[39, "Pos"] <- "SF"  #Harrison Barnes
nbaplayer_201819[103, "Pos"] <- "SG"  #Jimmy Butler
nbaplayer_201819[129, "Pos"] <- "SF"  #Wilson Chandler
nbaplayer_201819[379, "Pos"] <- "SG" #Kyle Kover
nbaplayer_201819[427, "Pos"] <- "C"  #Thon Maker
nbaplayer_201819[437, "Pos"] <- "SG" #Wesley Matthews
nbaplayer_201819[600, "Pos"] <- "SF" #Jonathon Simmons
nbaplayer_201819[611, "Pos"] <- "PF" #Jason Smith
nbaplayer_201819$Pos <- factor(nbaplayer_201819$Pos, levels = c("C", "PF", "SF", "SG", "PG"))
summary(nbaplayer_201819$Pos)
```


```{r }
ggplot(data = nbaplayer_201819, aes(x = Pos)) + geom_bar(aes(fill = Pos)) + theme_bw(base_size = 14) + 
    labs(title = "各位置人數", x = "場上位置", y = "球員人數") + 
    theme(text = element_text(family = "黑體-繁 中黑"))
table(complete.cases(nbaplayer_201819))
```

**若有NA值補0，考量實際狀況是整季無出手導致命中率無資料所致，不過後續也是取球員出場數>56場，故無影響後面的模型**
```{r}
nbaplayer_201819[is.na(nbaplayer_201819$FG.), "FG."] <- 0
nbaplayer_201819[is.na(nbaplayer_201819$X3P.), "X3P."] <- 0
nbaplayer_201819[is.na(nbaplayer_201819$X2P.), "X2P."] <- 0
nbaplayer_201819[is.na(nbaplayer_201819$eFG.), "eFG."] <- 0
nbaplayer_201819[is.na(nbaplayer_201819$FT.), "FT."] <- 0
table(complete.cases(nbaplayer_201819))
```

**因為球員球季中會被交易轉隊，故會有轉隊總和數據(TOT)，共計86位球員**  
```{r echo =TRUE}
length(nbaplayer_201819[nbaplayer_201819$Tm == "TOT", "Player"])

```

* **留下總和數據即可，因此刪除重複資料，資料處理後為530筆觀察資料，即530位不重複的球員，後續會再針對相關過高之變數以PCA降維度處理**  
* **反應變數*Y*為Pos、解釋變數*X*初步為24個**

```{r }
table(duplicated(nbaplayer_201819$Player))  #觀察名字為一值僅530筆，故重複了178筆
repindex <- which(duplicated(nbaplayer_201819$Player)) #返回索引值
nbaplayer_201819 <- nbaplayer_201819[-repindex, ] #刪除重複的球員名稱(轉隊)，留下總和資料
dim(nbaplayer_201819[, 7:ncol(nbaplayer_201819)])
names(nbaplayer_201819)
```


# 貳.Principal Component Analysis(PCA)

### 主要精神：

    1.將高維度資料降低維度，使用低維度資料表達，如：BMI指數、加權平均分數  
    2.找出「有效表達資料的新變數」，新變數為原變數的線性組合，並且捕捉最多的資料變異量*Variation  
    3.原解釋變數高度相關 *correlated*，有幾個解釋變數，則有幾個相對的主成份  

### 演算法(模型精神與參數估計)  

    1.每個主成份都是原始變數的加權平均  
    2.主成份間彼此不相關*(垂直)  
    3.越前面主成份解釋越多變異  
    4.拋棄較後面*PCA*進行降維  
  $$Y_{1} = a_{11}X_{1} + a_{12}X_{2} +...+  a_{1p}X_{p}$$
  $$Y_{2} = a_{21}X_{1} + a_{22}X_{2} +...+  a_{2p}X_{p}$$
  $$...$$
  $$Y_{p} = a_{p1}X_{1} + a_{p2}X_{2} +...+  a_{pp}X_{p}$$  
  
### 參數估計:估計$a_{ij}$達成兩大目標，保持不相關與極大化變異數
  $$Var(Y_{i}) = Var(a_{i1}X_{1} +...+ a_{ip}X_{p})  
    = a_{i1}^{2}Var(X_{1}) +...+ a_{ip}^{2}Var(X_{p}) + 
    2a_{i1}a_{i2}Cov(X_{1},X_{2}) + 2a_{ip-1}a_{ip}Cov(X_{p-1},X_{p})$$
    
    1.目標:極大化$Var(Y_{i})$  
    2.限制式:保持第i個主成份與前i-1個不相關  
    3.係數非負限制:非負每一個係數$a_{ij}$都需要≥0  
    4.係數稀疏限制:第i個主成份係數非零(第i個主成份係數a_i1...a_ip非零個數小於k) 

### *PCA*資料前處理   

    1.進行變數標準化  
    2.利用標準化變數估計主成份分析參數並得到主成份分數  

### *PCA*分析步驟  

  1. 每個主成份內的變數，若為負數則與該主成份互為反向關係，絕對值數字越大越不相關(貢獻越多)  
  2. 利用熱圖 *HeatMap* 視覺化權重矩陣  
  3. 利用點狀圖 *DotChart* 視覺化 *PC* 的權重  
  4. 利用「解釋變異量」決定要保留多少主成份    
      - 準則1:超過平均值  
      - 準則2:特定轉折處  
      - 準則3:解釋超過80%  
  5. 運用二元圖 *BitPlot* 理解原資料、變數與主成分的關係  
      - 每個點為所選取對應的主成份分數(降維、分群) 
      
***

### 實作過程  

#### 第一步:觀察各解釋變數相關性

```{r correlation_table}
kable(head(round(cor(nbaplayer_201819[, 8:ncol(nbaplayer_201819)]),3))) %>% 
  kable_styling() %>% row_spec(0, background = "red", color = "white") %>% scroll_box(height = "250px") 
```

#### 第二步:使用熱圖(heatmap)視覺化變數間的相關程度
1. 使用 ggplot2 繪製相關係數的熱圖  
2. 必須先將資料整理成 tidy 的「Var1 - Var2 - 相關係數」資料架構
3. 我們可以利用 reshape2套件中的melt函數輕鬆把矩陣格式轉換成 tidy 資料
4. 用geom_tile(繪製方形圖)繪製相關係數的熱圖囉！可以看到很清楚的變數群聚現象


```{r 熱圖觀察變項的相關程度}
#head(melt(cor(nbaplayer_201819[, 6:ncol(nbaplayer_201819)])),5)  

ggplot(melt(cor(nbaplayer_201819[, 8:ncol(nbaplayer_201819)])), 
       aes(x = Var1, y = Var2)) + geom_tile(aes(fill = value), colour = "white") + #這邊color為隔線顏色
    #scale_fill_gradient2為漸層顏色設定、midpoint為中點數值(default = 0)、mid為cor為0顏色
    scale_fill_gradient2(low = "navy", high = "tomato4", mid = "white", midpoint = 0) +
    #guides為設定圖例位置及顏色, 這邊如果寫colour = guide_legend(),圖例標題不會成功
    guides(fill = guide_legend(title = "Correlation")) +    
    theme_bw() + labs(title = "Correlations of Variables") +
    #調整相關座標軸、圖例位置的設定theme, legend.position = "bottom" 圖例位置會在下面
    theme(axis.text.x = element_text(angle = 90, hjust = 0.7, vjust = 1),  axis.title = element_blank())
```

#### 第三步：檢視變數間的相關性  

```{r}
#layout.exp 參數為針對變數標籤左側空出n個空白格; size為變量標籤大小;(nbreaks = 4, palette = "RdGy")需搭配使用
ggcorr(data = nbaplayer_201819[, 8:ncol(nbaplayer_201819)], geom = "blank",label = T, 
       hjust = 0.7, size = 3, layout.exp = 3, label_size = 3) +
    geom_point(size = 6, aes(color = coefficient > 0, alpha = abs(coefficient) >= 0.9)) + #框色大小
    scale_alpha_manual(values = c("TRUE" = 0.5, "FALSE" = 0)) + #標記的透明度
    guides(color = FALSE, alpha = FALSE)  

```

* 得分PTS與FG,FGA,2PA,PA...等相關  
* 失誤TOV得分PTS高度相關  
* 有效命中率(eFG%)與命中率高度相關，因其公式為命中率推導

**剔除相關係數為1之基礎變數:*FGA,X2PA,X3PA,TRB,FG*，整理資料後變數欄位剩下18個 **  
     
```{r}
nbaplayer_201819_V1 <- nbaplayer_201819[, -c(10,13,16, 20, 24)]
kable(nbaplayer_201819_V1[1:6, ]) %>% kable_styling() %>% row_spec(0, background = "orange") %>% scroll_box(height = "200px")
dim(nbaplayer_201819_V1[, 8:ncol(nbaplayer_201819_V1)])
```
 
* 觀察敘述統計量如下:

1.球員平均年紀在25歲，平均出賽時間近20分鐘．平均出賽場次在50場上下  
2.球員整體命中率中位數為43.85%，有效命中率為51.10%  
3.530位球員中，人數以SG居多、PG&PF次之、SF人數最少，不過聯盟中以SF的球員平均薪資最高    

```{r}
summary(nbaplayer_201819_V1[,3:ncol(nbaplayer_201819_V1)])
```

#### 第四步：建立非負稀疏主成份分析(Non-Negative Sparce PCA)  
1. 因主成分分析在解釋上會因為變數間有反向關係存在，使得解釋上有困難，故直接採用非負稀疏主成份分析  
2. 利用 *nsprcomp* 套件中的 *nscumcomp* 完成，其中有兩個重要的參數  
3. *k* 為非 0 係數個數，通常是「每個主成份期待非 0 係數個數」x 變數個數  
4. *nneg* 是否希望所有係數都非負，TRUE 代表有非負限制
5. 模型結果(輸出的值):
    + sdev表示每個PCA可以解釋的變異數為多少;  
    + rotation為變數的係數矩陣  
    + center 為標準化時每個變數的中心;  
    + scale 為標準化時每個變數的尺度;
    + x為利⽤標準化變數估計主成份分析的參數並得到主成份分數(score)  

```{r NSPCA model, echo = TRUE, results = "hide"}
set.seed(1234)
nspca.model <- nscumcomp(nbaplayer_201819_V1[, 8:ncol(nbaplayer_201819_V1)], nneg = T, scale. = T, k = 150)
```

**7個主成份可解釋8成以上的變異量**  

```{r}
summary(nspca.model) 
```

```{r, echo = TRUE }
var.exp <- tibble(
    pc = paste0("PC_", formatC(1:18, width = 2, flag = "0")),
    var = nspca.model$sdev^2,
    prop = (nspca.model$sdev)^2 / sum((nspca.model$sdev)^2),
    cum_prop = cumsum((nspca.model$sdev)^2 / sum((nspca.model$sdev)^2)))
kable(var.exp[1:7, ]) %>% kable_styling(row_label_position = "l") %>% row_spec(0, background = "navy", color = "white") %>% scroll_box(height = "250px", width = "500px")
```

***

**從互動圖表可觀察到：7個主成份解釋8成以上的變異量**

```{r, warning = FALSE}
plot_ly(
    x = var.exp$pc,
    y = var.exp$cum_prop,
    type = "bar") %>%
    layout(
        title = "Cumulative Proportion by Each Principal Component",
        xaxis = list(type = 'Principal Component', tickangle = -90),
        yaxis = list(title = 'Proportion'),
        margin = list(r = 30, t = 50, b = 70, l = 50)
    )
```

**非負稀疏主成份的係數權重。從熱圖中可以看到各主成份成分為**  

1.PC1 :「平均上場時間MP_index」  
2.PC2 :「二分球命中數X2P_index」  
3.PC3 :「失誤TOV_index」  
4.PC4 :「進攻籃板ORB_index」  
5.PC5 :「罰球FT._index」  
6.PC6 :「抄截STL_index」  
7.PC7 :「有效命中率攻eFG._index」  

```{r PC1}
# 使用dotchart，繪製主成份負荷圖
for (i in 1:7) {
  dotchart(nspca.model$rotation[,i][order(nspca.model$rotation[, i], decreasing=FALSE)] ,   # 排序後的係數
         main = paste("Loading Plot for PC", i, sep = ""),                      # 主標題
         xlab = "Variable Loadings",                         # x軸的標題
         col = rainbow(7))
}
                                       
```

```{r}

ggplot(melt(nspca.model$rotation[, 1:7]), aes(x = Var2, y = Var1)) + 
    geom_tile(aes(fill = value), colour = "gray") + #這邊color為隔線顏色
    #scale_fill_gradient2為漸層顏色設定、midpoint為中點數值(default=0)、mid為cor為0顏色
    scale_fill_gradient2(high = "navy", mid = "white", midpoint = 0) +
    #guides為設定圖例位置及顏色, 這邊如果寫colour = guide_legend(),圖例標題不會成功
    guides(fill = guide_legend(title = "Correlation")) + theme_bw() + 
    #調整相關座標軸、圖例位置的設定theme, legend.position = "bottom" 圖例位置會在下面
    theme(axis.text.x = element_text(angle = 0, hjust = 0.6, vjust = 1), axis.title = element_blank())

```

***

#### 第五步:這邊透過二元圖 BitPlot做各球員分析   

* 使用非負係數模型中的參數x，即轉置後主成份分數(score)  
```{r, echo=TRUE}
nspca.score <- data.frame(nspca.model$x)
row.names(nspca.score) <- nbaplayer_201819_V1$Player
```

* **球員個別分析(找出特別球員的方法，繪製「主成份分數」與「該主成份係數最大變數」的散佈圖**  
* **比如下圖可觀察，PC7有效命中率指標與球員得分比較**  
  + **固定PTS, 比較PC7:eFG._index--- **  
    * **James Harden得分效率極高，甚至遠高於鋒線球員，如:Joel Embiid、Giannis Antetokounmpo和LaBron James**  
  + **固定PC7:eFG._index, 比較PTS--- **  
    * **Paul George的平均得分高於Russel Westbrook，可知道相同的PC7下，前者得分效率較好，後者容易腦充血亂砍分**  
  
```{r}
plot_ly(
    x = round(nspca.score[, 7],3),
    y = nbaplayer_201819_V1$PTS,
    text = nbaplayer_201819_V1$Player,
    type = "scatter",
    mode = "markers") %>% 
    layout(
    title = "PTS VS. eFG.(PC_7) Score: Scatter Plot",
    xaxis = list(title = 'eFG._PC_7'),
    yaxis = list(title = 'PTS_index'),
    margin = list(r = 30, t = 50, b = 70, l = 50)
)
```

* **另外透過不同主成份的散佈圖，也可以找到多種面向都很傑出的球員如:平均上場時間(PC_1) VS.平均失誤(PC_3)**  
* **固定PC1:MP_index, PC3:TOV_index較低：比較Russell Westbrook與Kawhi Leonard，相同的平均上場時間，後者失誤數較少**  
* **固定PC3:TOV_index, PC1:MP_index較低：比較David Lillard與John Wall，差不多的平均失誤下，前者的出場時間較多，可見前者的穩定性較高**  

```{r}
plot_ly(
    x = nspca.score[, 1],
    y = nspca.score[, 3],
    text = nbaplayer_201819_V1$Player,
    type = "scatter",
    mode = "markers"
) %>% layout(
    title = "MP VS. TOV Score: Scatter Plot",
    xaxis = list(title = 'MP_index'),
    yaxis = list(title = 'TOV_index'),
    margin = list(r = 30, t = 50, b = 70, l = 50)
)
```

* 這邊發現一個現象就是在各主成份中，有太多重複的因子，導致特徵萃取後仍呈現相關，如:  
  + PC1 loading: MP, FT, X3P, FG.    
  + PC3 loading: TOV, X3P., FG., X3P  
* 後續再剔除變數上，將相關係數>0.7的變數初步做一次主成份，再跟原資料合併；或者透過Linear Regressiong篩選顯著變數後先行刪除不顯著變數  
```{r 檢視各球員 }
biplot(nspca.model, choices = c(1,3), col = c("tomato4", "navy"))
```

***

# 參.kNN分類

1. KNN (k nearest neighbor): 其定義為透過K個新數據(鄰居)，多數為某一分類，新數據即為其分類。  
2. 特徵：即正確判斷分類的屬性。  
3. 特徵樣本：由特徵值和正確分類所組成的集合，以便告訴機器給機器學習。  
4. 計算方式：找出與新樣本距離最近的K個特徵樣本，以Cosine Similarity為距離計算公式，即向量內積除以向量長度。  

 $$ similarity(A,B)= \frac{A*B}{||A||*||B||} =\frac{\sum^{n}_{i=1}(A_i * B_i)} {\sqrt\sum_{i=1}^n (A_i)*\sqrt\sum_{i=1}^n (B_i)}$$
5. 優點為簡單不需要輸入資料假設，對於異常值不敏感;缺點為計算量大，非常耗時且占記憶體空間大。可探討的議題有「參數k值如何選取,這裡的k是鄰近的點」，如：資料數的平方根、「如何提升效能」。及使用PCA後的資料

***

### 分析資料與問題  

* Problem：透過籃球基礎數據，分類球員位置並且檢視分類正確率
* Data：NBA 2018-19賽季資料
* 目標變數(1)：Pos
* 最終預測變數(18)："MP" "FG" "FG." "X3P" "X3P." "X2P" "X2P." "eFG." "FT" "FT." "ORB" "DRB" "AST" "STL" "BLK" "TOV" "PF" "PTS" 

### 分析實作步驟  

1. 資料載入與檢視  
2. 資料前處理  
    + 將兩位置球員，如：PF-SF改為單一位置  
    + 針對變數:X3P, FG., X2P, eFG., FT.的NA值補0  
    + 刪除轉隊球員資料，僅留下整季的資料  
    + 刪除相關性為1的預測變數:FGA,X2PA,X3PA,TRB,FG(為所有狀況的合計命中數)
3. 建立kNN分類模型
4. 結碖

***

### 資料載入與檢視

**建立kNN模型第三個參數:training set和testing set 的正確分類，以便後續模型和混淆矩陣使用**  
```{r  第三個參數, echo = T}
pos.kNN <- nbaplayer_201819_V1[nbaplayer_201819_V1$G >= 56 , 3] 
nbaplayer_201819_V2.test <- nbaplayer_201819_V1[nbaplayer_201819_V1$G >= 56, -c(1:5)] %>% scale() %>% round(3)
```

**依序建立kNN模型的三個參數資料，以配適kNN模型中**   
```{r 設定隨機種子以固定70%訓練集, echo = T}
set.seed(168)
index.kNN <- sample(nrow(nbaplayer_201819_V2.test), 0.7 * nrow(nbaplayer_201819_V2.test))
train.set.kNN <- nbaplayer_201819_V2.test[index.kNN, ] #training set KNN第一個參數
test.set.kNN <- nbaplayer_201819_V2.test[-index.kNN, ] #test set KNN第二個參數
train.pos.kNN <- pos.kNN[index.kNN]           #training set正確答案，KNN第三個參數
test.pos.kNN <- pos.kNN[-index.kNN]           #真實答案，作為後面混淆矩陣對照 (這邊也要注意格式轉換，目前仍為tibble)
```


**另外為找出最適k值，使用*Elbow Method*來選擇最適合的k值**  

```{r}
predicted.pos = NULL
error.rate.kNN <- numeric(0)
set.seed(500)
for (i in 1:20) {
    predicted.pos.kNN <- knn(train.set.kNN, test.set.kNN, as.matrix(train.pos.kNN), k = i)
    error.rate.kNN[i] <- mean(as.matrix(test.pos.kNN) != predicted.pos.kNN)
}
round(error.rate.kNN,3)
k.values.kNN <- 1:20
error.tb.kNN <- tibble(k.values.kNN, error.rate.kNN)
ggplot(error.tb.kNN, aes(x = k.values.kNN, y = error.rate.kNN)) + geom_point(size = 0.8) + 
    geom_text(aes(label = k.values.kNN), vjust = 3, size = 2, colour = "red") +
    geom_line(lty = "dotted", colour = 'blue')

# set.seed(500)
# predicted.pos.kNN <- knn(train.set.kNN, test.set.kNN, as.matrix(train.pos.kNN), k = 8) 
# postable.kNN <- table(test.pos.kNN, factor(predicted.pos.kNN, levels = c("C", "PF", "SF", "SG", "PG")), 
#                   dnn = c("預測", "實際"))
# accuracy.kNN <- sum(diag(postable.kNN))/sum(postable.kNN) 
# kable(postable.kNN) %>% kable_styling() %>% row_spec(0, background = "red") %>% scroll_box(height = "250px")
# plot(t(postable.kNN), main = "Confusion Matrix for Pos Clustering", 
#      xlab = "Pos", ylab = "Cluster", col = rainbow(5))
# 
# apply(postable.kNN, 2, sum)
# 
# cat("準確率",round(accuracy.kNN * 100,3), "%")
```

**觀察圖形得到當k值為8 (鄰近的點取量), Error.Rate為最低43.2%，下面建立Confusion Matrix**
```{r}
set.seed(500)
predicted.pos.kNN <- knn(train.set.kNN, test.set.kNN, as.matrix(train.pos.kNN), k = 8)
postable.kNN <- table(test.pos.kNN, factor(predicted.pos.kNN, levels = c("C", "PF", "SF", "SG", "PG")),
                  dnn = c("預測", "實際"))
accuracy.kNN <- sum(diag(postable.kNN))/sum(postable.kNN)
kable(postable.kNN) %>% kable_styling() %>% row_spec(0, background = "red") %>% scroll_box(height = "250px")
plot(t(postable.kNN), main = "Confusion Matrix for Pos Clustering",
     xlab = "Pos", ylab = "Cluster", col = rainbow(5))
apply(postable.kNN, 2, sum)
cat("準確率",round(accuracy.kNN * 100,3), "%")

```

### 結論：

* 最終得到正確率58.025%(全部資料為56.604%)，顯示剔除出場數(G)低於56場資料，可提到分類準確率
  + 特徵樣本的好壞和樣本當中分類的數量影響分類結果準度，若A類數量大於B分類，可預期K個最近距離鄰居高機率分類為A，分類失去準度；  
  + 實務上SF這個位置有10個人被分為PF，表示這位置選手可能很多事PF兼打、位置SF分類不明確，主要其位置介於SG, PF之間，且該位置球員資料較少，導致計算機在解讀時較不明確




<!-- **因前面PCA部分已得到主成份分數(score)，故後續僅需針對kNN模型相關參數匯入資料建模即可** -->

<!-- **取7個主成份分數作為新資料如下，這邊再註明一下取七個主成份**   -->
<!-- ```{r 以PCA所得到的主成分分數作為kNN建模分類依據 nbaplayer_kNN(PCA的資料)} -->
<!-- nbaplayer_kNN <- cbind(nbaplayer_201819_V1[, 2:3], nspca.score[, 1:7]) -->

<!-- kable(round(nbaplayer_kNN[1:10, 3:ncol(nbaplayer_kNN)],3)) %>% kable_styling(bootstrap_options = "condensed") %>% row_spec(0, background = "springgreen") %>% scroll_box(height = "200px", width = "700px") -->
<!-- ``` -->

<!-- ### 建立kNN分類模型 -->
<!-- `knn(train, test, cl, k = 1, l = 0, prob = FALSE, use.all = TRUE)` -->

<!-- * **建立kNN模型第三個參數:training set和testing set 的正確分類，以便後續模型和混淆矩陣使用**   -->
<!-- ```{r , echo = T} -->
<!-- pos <- nbaplayer_kNN[, 2] #產製單一分類變數train.pos作為後續KNN model的第三個參數樣本 -->
<!-- ``` -->

<!-- **依序建立kNN模型的三個參數資料，以配適kNN模型中**    -->
<!-- ```{r 設定隨機種子以固定70%訓練集, echo = T} -->
<!-- set.seed(168) -->
<!-- index <- sample(nrow(nbaplayer_kNN), 0.7 * nrow(nbaplayer_kNN)) -->

<!-- train.set <- nbaplayer_kNN[, -c(1,2)][index, ] #training set KNN第一個參數 -->

<!-- test.set <- nbaplayer_kNN[, -c(1,2)][-index, ] #test set KNN第二個參數 -->

<!-- train.pos <- pos[index]           #training set正確答案，KNN第三個參數 -->
<!-- test.pos <- pos[-index]           #真實答案，作為後面混淆矩陣對照 (這邊也要注意格式轉換，目前仍為tibble) -->
<!-- ``` -->

<!-- **另外為找出最適k值，使用*Elbow Method*來選擇**   -->
<!-- ```{r 找最佳k值} -->
<!-- predicted.pos = NULL -->
<!-- error.rate <- numeric(0) -->
<!-- for (i in 1:20) { -->
<!--     set.seed(168) -->
<!--     predicted.pos <- knn(train.set, test.set, as.matrix(train.pos), k = i) -->
<!--     error.rate[i] <- mean(as.matrix(test.pos) != predicted.pos) -->
<!-- } -->
<!-- print(error.rate) -->
<!-- k.values <- 1:20 -->
<!-- error.tb <- tibble(k.values, error.rate) -->
<!-- ggplot(error.tb, aes(x = k.values, y = error.rate)) + geom_point(size = 0.8) +  -->
<!--     geom_text(aes(label = k.values), vjust = 3, size = 2, colour = "tomato") + -->
<!--     geom_line(lty = "dotted", colour = 'navy') -->
<!-- ``` -->

<!-- * **觀察圖形得到當k值為17, error.rate皆為最低57.86%，接著建立混淆矩陣Confusion Matrix來檢視分類的正確率**   -->
<!-- ```{r} -->
<!-- predicted.pos <- knn(train.set, test.set, as.matrix(train.pos), k = 17)  -->
<!-- postable <- table(test.pos, factor(predicted.pos, levels = c("C", "PF", "SF", "SG", "PG")), dnn = c("預測", "實際")) -->
<!-- accuracy <- sum(diag(postable))/sum(postable)  -->
<!-- kable(postable) %>% kable_styling() %>% row_spec(0, background = "gray")%>% scroll_box(height = "250px") -->
<!-- plot(t(postable), main = "Confusion Matrix for Pos Clustering",  -->
<!--      xlab = "Pos", ylab = "Cluster", col = c("black", "pink2", "orange2", "navy", "brown")) -->
<!-- apply(postable, 2, sum) -->
<!-- accuracy -->

<!-- ``` -->

*** 

# 肆.Hierarchical Cluster分群

<!-- **過去共計做了五次階層分析Hierarchical Clustering，本次報告僅以第五次為主** -->

<!-- 1. **第一次原資料執行，準確率17.16%**   -->
<!-- 2. **第二次將相關性過高欄位變數刪除，X2PA,X3PA,FTA,TOV,PF (準確率為15.08%)** -->
<!-- 3. **第三次清洗取出場數超過27場(398筆資料)球員資料(準確率為19.84%)**   -->
<!-- 4. **第四次先行對>=27場(球員資料做PCA,在做hclust(準確率為21.3%)**   -->
<!-- 5. **第五次實作球員出場數>56場(中位數)，並且做PCA(準確率為21.509%)**   -->

* 階層分群可被用運用數值與類別資料  
  + d:由dist()函數計算出來資料兩兩間的相異度矩陣(dissimilarity matrix)，即兩兩資料間的距離矩陣
  + method:群(clusters)聚合或連結的方式。包括：single(單一）、complete（完整）、average（平均)、Ward’s（華德）和 centroid（中心）等法。其中又以average(平均)聚合方法被認為是最適合的。不同方法對階層分群結果亦有極大影響。  
* 由於階層式分群是依據個體間的「距離」來計算彼此的相似度。  
  + 我們會先使用dist()函數，來計算所有資料個體間的「距離矩陣(Distance Matrix)」  
  + 「距離」的算法又有：(1)歐式距離(2D)(2)曼哈頓距離(1D)
* 資料分群屬於非監督式學習法(Unsupervised Learning)，即資料沒有標籤(unlabeled 
data)或沒有標準答案，無法透過所謂的目標變數(response variable)來做分類之訓練。也因為資料沒有標籤之緣故，與監督式學
習法和強化式學習法不同，非監督式學習法無法衡量演算法的正確率。



### 分析資料與問題
**Problem：透過籃球基礎數據，針對球員位置標籤做分群並且檢視分群正確率**  
**Data：NBA 2018-19賽季資料**  
**目標變數Y(1)：Pos**  
**最終預測變數X(7)：第貳部分非負係數主成份分析所得到的scores:PC1-PC7 ** 

**前面非負係數主成分維度為530筆資料及前七個主成份作為後續分群依據**
<!-- **取出出賽場次高於40場之球員數據做分析** -->
```{r}
dim(nspca.score[, 1:7]) 
```


**使用新的資料作為階層式分群依據，並且使用歐式距離計算相異度矩陣**
```{r}
#inputdata.hc <- cbind(nbaplayer_201819_V1$G, nspca.score[, 1:7])
#inputdata.hc <- cbind(nbaplayer_201819_V1$G > 40, nspca.score[, 1:7])
#inputdata.hc <- nbaplayer_201819_V1[nbaplayer_201819_V1$G > 56, ] 後續修改

inputdata.hc <- cbind(nbaplayer_201819_V1[, c(3,6)], nspca.score[, 1:7])

inputdata.hc <- inputdata.hc[inputdata.hc$G >= 56, ]
E.dist.report <- dist(x = inputdata.hc, method = "euclidean")
```

**將以上資料間距離作為參數投入階層式分群函數：`hclust()`, 參數預設為`method = "complete"`)**  
**使用歐式距離進行分群**
```{r, include=F}
set.seed(500)
h.E.cluster.report <- hclust(E.dist.report) # 預設 method = "complete",
plot(h.E.cluster.report, xlab= "完整聚合演算法", family="黑體-繁 中黑")
```

** 採用歐式距離搭配華德最小變異聚合演算法。可透過設定`hclust()`參數`method = ”ward.D2"` **
```{r, include=FALSE,echo=TRUE}
set.seed(500)
h.E.Ward.report <- hclust(E.dist.report, method = "ward.D2")
#plot(h.E.Ward.report, main = "Dendrogram of hclust",
#     xlab = "華德法: Ward's Method", col = "navy", family="黑體-繁 中黑")  # 華德法
```

* 因`agnes()`函數可計算聚合係數 *agglomerative coefficient*。    
* 聚合係數是衡量群聚結構被辨識的程度，聚合係數越接近1代表有堅固的群聚結構*strong clustering structure*  
* 數據顯示以華德法*Ward's*配合歐式距離所分群效果較佳，其聚和係數0.992為最高
```{r}
set.seed(500)
hc2.report <- agnes(E.dist.report, method = "ward") # agnes()函數預設是method ="average", 共計六種演算法

hc2.report$ac 
m <- c("average", "single", "complete", "ward", "weighted","gaverage")
names(m) <- c("average", "single", "complete", "ward", "weighted","gaverage")

ac.report <- function(x){
    agnes(E.dist.report, method = x)$ac
}
map_dbl(m, ac.report) #顯示以華德法配合歐式距離所分群效果較佳

#agnes()產生的樹狀圖繪出可使用函數pltree()
#set.seed(500)
#hc2.report <- agnes(E.dist.report, method = "ward")
pltree(hc2.report, cex = 0.6, hang = -1, main = "Dendrogram of agnes", 
       col = "orange",xlab = "華德法: Ward's Method", family="黑體-繁 中黑")
```

* 對階層分群結果產生的樹狀結構進行截枝可以將觀測值分成數群。截枝方法有兩種：
* 指定所要的分群數: `rect.hclust(k=…)、cutree(k=…)`
* 指定截枝的位置: `rect.hclust(h=…)`
* 使用`hclust()函數`搭配華德法`method = "ward.D2"`(華德法有較高的聚合係數)  

*** 

* **指定分群為5組和6組**
```{r}
#h.E.Ward.report <- agnes(E.dist.report, method = "ward")
#par(mfrow = c(1, 1))
plot(h.E.Ward.report, main = "Dendrogram of hclust", xlab = "華德法: h.E.Ward's Method", col = "navy", family="黑體-繁 中黑")
rect.hclust(h.E.Ward.report, k = 5, border = "black")
rect.hclust(h.E.Ward.report, k = 6, border = "tomato4") 
```

<!-- **如果要將資料標記上分群結果，可使用cutree()，並指定參數k為所欲截枝群樹**   -->

```{r , results = "hide"}
cut.E.Ward.report <- cutree(tree = h.E.Ward.report, k = 5)
kable(cut.E.Ward.report[1:6], "html") %>% kable_styling(bootstrap_options = "condensed", row_label_position = "l",position = "left") %>% row_spec(0, background = "red")%>% scroll_box(height = "200px", width = "500px")
```

<!-- **我們使用`factoextra`套件中的`fviz_clusterc`函數來將分群結果視覺化，可以看到兩個主成分解釋了80%以上變異** -->
**檢視比較分群結果**
```{r}
#fviz_cluster(list(data = inputdata.hc, cluster = cut.E.Ward.report),ellipse = F, labelsize = 9)

hcPosTable.report <- table(cut.E.Ward.report, factor(inputdata.hc$Pos, levels = c("C", "PF", "SF", "SG", "PG")), dnn = c("Predict", "Actual"))

plot(t(hcPosTable.report), main = "Confusion Matrix for Pos Clustering", 
     xlab = "Pos", ylab = "Cluster", col = rainbow(5))
#accuracy <- sum(diag(hcPosTable.report))/sum(hcPosTable.report) * 100
#cat("階層式分群準確率為", round(accuracy,3), "%") 
```

**可以看到抄截(STL)和火鍋(BLK)散佈圖，較能區分球員位置(Pos)**  
**其餘scatter皆較無法分出各位置，也呼應分群效果不佳的主要因素**  

```{r}
ggplot(data = nbaplayer_201819_V1, mapping = aes(x = STL, y = BLK)) + 
  geom_point(mapping = aes(col = Pos))
ggplot(data = nbaplayer_201819_V1, mapping = aes(x = TOV, y = BLK)) + 
  geom_point(mapping = aes(col = Pos))
ggplot(data = nbaplayer_201819_V1, mapping = aes(x = ORB, y = BLK)) + 
  geom_point(mapping = aes(col = Pos))
ggplot(data = nbaplayer_201819_V1, mapping = aes(x = ORB, y = AST)) + 
  geom_point(mapping = aes(col = Pos))

```

### 結論:  
<!-- 1.將全部資料放入分群模型中，分群準確率為20%   -->
<!-- 2.篩選出賽場次超過賽季一半41場之球員，分群準確率提高至19.811%   -->
<!-- 3.篩選出賽場次超過中位數56場之球員，分群準確率提高至至23.048%    -->
1. 這邊可以觀察到，非監督式學習法因為在去標籤化、亦無標準答案下做分類訓練，造成數據分群不明顯，當然實際狀況是因為現在有許多國際球員打法較為全面，多數全員分工狀況較早期不明顯，內線球員可投外線，鋒衛球員亦可衝搶籃板。  
2. 顯示以出賽場次(G)作為分群依據，有效提高分群結果。後續將實作切割式分群(K-Means & 
  K-Medoid)，檢視分群效果並和Hierarchical Cluster比較  

***

# 伍.K-Means

> 1.Partitional Clustering, 切割式分群，屬於資料分群屬的一種方法。  
> 2.資料分群亦屬於非監督式學習，所處理的資料是沒有正確答案/標籤/目標變數可參考的。  
> 3.常見的切割式分群演算法包括kMeans(群內平均值為中心點), kMedoid(群內中位數為中心點)。


### K-Means  

>1.分群演算法中最出名的即為K-Means演算法。  
 2.K-Means會根據一些距離的測量將觀測值分成數個組別。  
 3.需要事前指定分群數目。  
 4.目的是將極大化群內相似度，和最大化群間相異度。  
 5.指定群聚的平均值作為中心點(centroid)。  
 6.投入變數以連續變數為佳（kmeans(x = …只能為數值矩陣…)）
 7.同階層式分群是讓子集間差異最大化，而子集內差異最小化

**實作過程步驟如下:**  

  
Step 1: 資料準備(使用前面主成份分析後的score矩陣)  
Step 2: 衡量群聚距離(歐式距離)
Step 3: 找最佳k值(Elbow Method法)
Step 4: K-Means分群  
Step 5: K-Medoid分群  

延續階層式分群資料(主成份分數資料)，執行*K-Means*、*K-Medoid*  

### 最佳*k值、nstart*兩參數*(Elbow Method法)*  
 
1. 先找最佳*nstart*參數  
* 因實際資料位置為5(即k=5),可計算出群聚內變異加總在tot.withinss約在n > 30時出現趨於穩定。故取參數`nstrt = 30`即可
```{r}
set.seed(123)
wss.report <- function(n) {
    kmeans( x = nspca.score[,1:7], centers =  5, nstart = n )$tot.withinss
}
#Compute and plot wss for k = 1 to k = 15
n.values.report <- seq(1, 100)
# extract wss for 2-15 clusters
wss_values.report <- map_dbl(n.values.report, wss.report)
plot(n.values.report, wss_values.report,
     type = "b", pch = 19, frame = FALSE, 
     xlab = "Number of nstart n",
     ylab = "Total within-clusters sum of squares")

```

找最佳*k*值   

* 在決定執行初始中心點30下(nstart = 30)，使用*Elbow Method*法，找出資料分為3群(k=3)時出現轉折為較佳  
```{r}
set.seed(123)
# function to compute total within-cluster sum of square 
wssv2.report <- function(k) {
    kmeans( x = nspca.score[,1:7], centers =  k, nstart = 25)$tot.withinss
}

# Compute and plot wss for k = 1 to k = 15
k.values.report <- 1:15

# extract wss for 2-15 clusters
wss_valuesv2.report <- map_dbl(k.values.report, wssv2.report)

plot(k.values.report, wss_valuesv2.report,
     type = "b", pch = 19, frame = FALSE, 
     xlab = "Number of clusters K",
     ylab = "Total within-clusters sum of squares")

```

* 決定好分群數後，執行kmeans函數，並且視覺化，這邊使用factoextra套件的`fviz_cluster()`函數
* 由於模型資料已是標準化後的值，故參數 `stand=F`，另外可以看到第一個主成份以解釋近90%變異，從球員名稱大約推測橫軸為主要球隊核心隊員(得分高、上場時間高)，縱軸Y為內線球員  
```{r}
set.seed(500)
k_clust5.report <- kmeans(inputdata.hc[, -c(1,2)], centers = 3, nstart = 25)
#k_clust5.report <- kmeans(nbaplayer_201819_V2[, 3: ncol(nbaplayer_201819_V2)], centers = 3, nstart = 80)


 # kmeans_clust_table5_report <- table(k_clust5.report$cluster, factor(inputdata.hc$Pos, levels = c("C", "PF",  "SF","SG","PG")), dnn = c("Predict", "Actual"))
 #  accuracy.kmeans5.report <- sum(diag(kmeans_clust_table5_report))/sum(kmeans_clust_table5_report) * 100
 #  cat("K-Means分群準確率", round(accuracy.kmeans5.report, 3), "%")

#本身已標準化資料，故參數stand = F, 大致檢視圖型前二個主成份為X2P_index,MP_index
#rownames(nbaplayer_201819_V2) <- nbaplayer_201819_V1$Player  (處理過一次即可)
fviz_cluster(k_clust5.report, inputdata.hc[, -c(1,2)], labelsize = 9, stand = F, ellipse = F, main = "K-Means")
#autoplot(object = kmeans(nspca.score[, 1:7], 3, nstart = 25), data= nspca.score)

```

* 第二張圖為使用傳統的成對資料(分類結果,原始資料)，以前標記資料位置來繪製散佈圖來檢視分群結果
* 前面在階層式分群時，由於進攻籃板和抄截散佈圖，較能區分各位置，
* 前面PCA之二元圖有提到，PG這邊點為James Harden,因此推測下圖PCA成分橫軸以是PTS.MP，縱軸是TOV成份。第一群以後衛群為主、主第二群以內線球員為主、第三群較為平均
```{r 同autoplot結果nspca.score[, 1:7]}
inputdata.hc[, -c(1,2)] %>%
    as_tibble() %>%
    mutate(cluster = k_clust5.report$cluster, #新增分群結果
           Pos = inputdata.hc$Pos#nbaplayer_201819_V1$Pos #將列名稱指定為原始資料標籤
    ) %>%
    # 參數check_overlap = T可刪除重複資料，以便觀察各群的內容
    ggplot(aes(PC2, PC1, color = factor(cluster),label = Pos)) + 
    geom_text(show.legend = T, check_overlap = T)+ labs(x = "X2P_index", y = "MP_index")


```

嘗試多種k的分群效果
```{r echo = T}
set.seed(500)
k_clust_2.report <- kmeans(inputdata.hc[, -c(1,2)], centers = 2, nstart = 25)
k_clust_3.report <- kmeans(inputdata.hc[, -c(1,2)], centers = 3, nstart = 25)
k_clust_4.report <- kmeans(inputdata.hc[, -c(1,2)], centers = 4, nstart = 25)
k_clust_5.report <- kmeans(inputdata.hc[, -c(1,2)], centers = 5, nstart = 25)

```


比較分成各群的視覺化圖表，但似乎無看看出哪個分群較佳
```{r echo = T}
p1.report <- fviz_cluster(k_clust_2.report, geom = "point", stand = F, ellipse = F,data = inputdata.hc[, -c(1,2)]) + ggtitle("k = 2")

p2.report <- fviz_cluster(k_clust_3.report, geom = "point", stand = F, ellipse = F,data = inputdata.hc[, -c(1,2)]) + ggtitle("k = 3")
p3.report <- fviz_cluster(k_clust_4.report, geom = "point", stand = F, ellipse = F,data = inputdata.hc[, -c(1,2)]) + ggtitle("k = 4")
p4.report <- fviz_cluster(k_clust_5.report, geom = "point", stand = F, ellipse = F, data = inputdata.hc[, -c(1,2)]) + ggtitle("k = 5")
grid.arrange(p1.report, p2.report, p3.report, p4.report, nrow = 2) # Arrange multiple grobs on a page (將不會影響到par()中參數設定)
```

***

# 陸.K-Medoid

> *最常使用的演算法為PAM(Partitioning Around Medoid, 分割環繞物件法)  
>  *K-Medoids比K-Means更強大之處在於他最小化相異度加總值，而非僅是歐式距離平方和  
> 使用K-Mediods演算法時，中心點將選選擇群內某個觀測值，而非群內平均值，就像中位數一樣，較不易受離群值所影響。是K-Means更強大的版本  

(依情況決定是否將Step 4的kmeans()法取代，這邊跳過不展示)

* **使用cluster套件中的pam()函數,幾個參數如下**
  + x:可為數字矩陣、data.frame、dissimilraity matrix  
  + 若為dissimalirity matrix，可以是透過daisy()或dist()計算個體間距離的結果  
    - 其中daisy()可以處理類別行變數的距離矩陣計算，須設定參數metric = c(“gower”)
    - dist()則可以指定使用method = “euclidean” 或 “manhattan”。
    
***

# 柒.決定最適分群數目

最大目的:使群內的總變異最小；使群間的總變異最大(SSE)  

1.Elbow Method（手肘法)
2.Average Silhouette method（平均側影法輪廓係數法）



### Elbow Method

* 使用factoextra套件中的`fviz_nbclust()`函數，簡化上述流程,找出曲線彎曲（如膝蓋彎曲）處對應的k值，即各群群內變異加總值趨於收斂的轉折點，作為最佳的分群數目
* 最小化各群群內變異加總 （total within-cluster variation 或 total within-cluster sum of square，簡稱wss)

* 這裡使用的是`hcut()`，屬於`factoextra`套件，並非上面提的`hclust()`
```{r echo=T}

fviz_nbclust(inputdata.hc[, -c(1,2)], 
             FUNcluster = hcut,  # hierarchical clustering
             method = "wss",     # total within sum of square
             ) + 
labs(title = "Elbow Method for HC") +
    
geom_vline(xintercept = 3,       # 在 X=3的地方 
           linetype = 4)         # 畫一條虛線
```

```{r}
set.seed(500)
fviz_nbclust(x = inputdata.hc[, -c(1,2)], FUNcluster = pam, method = "wss") + labs(title="Elbow Method for K-Medoid") +
geom_vline(xintercept = 3,       # 在 X=3的地方 
           linetype = 3)         # 畫一條虛線

```


### Average Silhouette Method

* 該方法是檢視各分群一致性的品質，每個分群$C_1$ ~ $C_k$都有對應的Silhouette width，可由歐式距離或者曼哈頓距離計算  
* 每個群聚中，各物件 [Silhouette width](https://en.wikipedia.org/wiki/Silhouette_(clustering)) 衡量該物件是否歸類在合適得群聚中,其數值介於−1 to +1 
  + Silhouette width為正或數值很大，則表示該觀測值被分派到合適群聚  
  + Silhouette width為負或數值很小，則表示該觀測值分群結果不佳  
* 如果許多點都是負值或很低，表示分群太少或者太多，公式定義如下

$$s(i)=\frac{b(i)-a(i)} {max\{a(i),b(i)\}}$$

* 其中
  + a(i) = 資料點(i)，它與群內其他資料點的平均距離  
  + b(i) = 資料點(i)，它與其它群內資料點的平均距離取最小值  
  + s(i) = 側影係數，可以視為資料點(i)，在它所屬的群內是否適當的指標  
  
  

```{r ,include=FALSE}
ss <- silhouette(kmeans(x = inputdata.hc[, -c(1,2)], centers = 3, nstart = 25)$cluster, dist(inputdata.hc[, -c(1,2)]))
plot(ss)
```

```{r}
fviz_nbclust(x = inputdata.hc[, -c(1,2)], FUNcluster = hcut, method = "silhouette", linecolor = "navy") + labs(title="Avg.Silhouette Method for HC")
```


```{r}
fviz_nbclust(x = inputdata.hc[, -c(1,2)], FUNcluster = kmeans, method = "silhouette", linecolor = "navy") + labs(title="Avg.Silhouette Method for K-Means")
```


```{r}
set.seed(500)
fviz_nbclust(x = inputdata.hc[, -c(1,2)], FUNcluster = pam, method = "silhouette", linecolor = "tomato4") + labs(title="Avg.Silhouette Method for K-Medoid")
```


### 結論:  

> 1. 因分群為非監督式學習，因此初步方向可先依據資料特性、分群目的綜合考量不同方式，決定分群群數來進行資料分析  
> 2. 不同的分群數目，往往會對後面的分析有不同影響 
> 3. 觀察到使用Average Silhouette Method，分群數為2，經判斷及查找資料，推測與原始資料分布型態有關。
> 4. 初步執行PCA時，可先透過一套理論根據篩選顯著變數，如:Linear Regression的逐步迴歸篩選顯著變數  




