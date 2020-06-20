<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="classifier.classification_report">classifier.classification_report<a class="anchor-link" href="#classifier.classification_report">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p><b>Definition for Multiclass Classifier:</b></p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">classifier</span><span class="o">.</span><span class="n">classification_report</span><span class="p">(</span><span class="n">cutoff</span> <span class="o">=</span> <span class="p">[],</span>
                                 <span class="n">labels</span><span class="p">:</span> <span class="nb">list</span> <span class="o">=</span> <span class="p">[])</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p><b>Definition for Binary Classifier:</b></p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">classifier</span><span class="o">.</span><span class="n">classification_report</span><span class="p">(</span><span class="n">cutoff</span><span class="p">:</span> <span class="nb">float</span> <span class="o">=</span> <span class="mf">0.5</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Computes a classification report using multiple metrics to evaluate the model (AUC, accuracy, PRC AUC, F1...). In case of multiclass classification, it will consider each category as positive and switch to the next one during the computation.</p>
<h3 id="Parameters">Parameters<a class="anchor-link" href="#Parameters">&#182;</a></h3><table id="parameters">
    <tr> <th>Name</th> <th>Type</th> <th>Optional</th> <th>Description</th> </tr>
    <tr> <td><div class="param_name">cutoff</div></td> <td><div class="type">float / list </div></td> <td><div class = "yes">&#10003;</div></td> <td>Cutoff for which the tested category will be accepted as prediction. In case of multiclass classification, each tested category becomes the positives and the others are merged into the negatives. The list will represent the classes threshold. If it is empty or invalid, the best cutoff will be used.</td> </tr>
    <tr> <td><div class="param_name">labels</div></td> <td><div class="type">list</div></td> <td><div class = "yes">&#10003;</div></td> <td>List of the different labels to be used during the computation.</td> </tr>
</table>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Returns">Returns<a class="anchor-link" href="#Returns">&#182;</a></h3><p><a href="../../../utilities/tablesample/index.php">tablesample</a> : An object containing the result. For more information, check out <a href="../../../utilities/tablesample/index.php">utilities.tablesample</a>.</p>
<h3 id="Example">Example<a class="anchor-link" href="#Example">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[11]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Multiclass Classification</span>
<span class="kn">from</span> <span class="nn">vertica_ml_python.learn.ensemble</span> <span class="k">import</span> <span class="n">RandomForestClassifier</span>
<span class="n">model</span> <span class="o">=</span> <span class="n">RandomForestClassifier</span><span class="p">(</span><span class="n">name</span> <span class="o">=</span> <span class="s2">&quot;public.RF_iris&quot;</span><span class="p">,</span>
                               <span class="n">n_estimators</span> <span class="o">=</span> <span class="mi">20</span><span class="p">,</span>
                               <span class="n">max_features</span> <span class="o">=</span> <span class="s2">&quot;auto&quot;</span><span class="p">,</span>
                               <span class="n">max_leaf_nodes</span> <span class="o">=</span> <span class="mi">32</span><span class="p">,</span> 
                               <span class="n">sample</span> <span class="o">=</span> <span class="mf">0.7</span><span class="p">,</span>
                               <span class="n">max_depth</span> <span class="o">=</span> <span class="mi">3</span><span class="p">,</span>
                               <span class="n">min_samples_leaf</span> <span class="o">=</span> <span class="mi">5</span><span class="p">,</span>
                               <span class="n">min_info_gain</span> <span class="o">=</span> <span class="mf">0.0</span><span class="p">,</span>
                               <span class="n">nbins</span> <span class="o">=</span> <span class="mi">32</span><span class="p">)</span>
<span class="n">model</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="s2">&quot;public.iris&quot;</span><span class="p">,</span> <span class="p">[</span><span class="s2">&quot;PetalLengthCm&quot;</span><span class="p">,</span> <span class="s2">&quot;PetalWidthCm&quot;</span><span class="p">],</span> <span class="s2">&quot;Species&quot;</span><span class="p">)</span>
<span class="c1"># Multiclass Classification: Using a fixed cutoff</span>
<span class="n">model</span><span class="o">.</span><span class="n">classification_report</span><span class="p">(</span><span class="n">cutoff</span> <span class="o">=</span> <span class="mf">0.33</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Iris-setosa</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Iris-versicolor</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Iris-virginica</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>auc</b></td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">0.9962000000000002</td><td style="border: 1px solid white;">0.997</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>prc_auc</b></td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">0.9928891891647207</td><td style="border: 1px solid white;">0.9941765474895515</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>accuracy</b></td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">0.9533333333333334</td><td style="border: 1px solid white;">0.9666666666666667</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>log_loss</b></td><td style="border: 1px solid white;">0.0095578366458892</td><td style="border: 1px solid white;">0.0382878575678393</td><td style="border: 1px solid white;">0.036209467180686</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>precision</b></td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">0.96</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>recall</b></td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">0.8771929824561403</td><td style="border: 1px solid white;">0.9411764705882353</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>f1_score</b></td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">0.9345794392523363</td><td style="border: 1px solid white;">0.9600989792762139</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>mcc</b></td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">0.9032106474595007</td><td style="border: 1px solid white;">0.9254762227411247</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>informedness</b></td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">0.8771929824561404</td><td style="border: 1px solid white;">0.9209744503862152</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>markedness</b></td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">0.9300000000000002</td><td style="border: 1px solid white;">0.9299999999999999</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>csi</b></td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">0.8771929824561403</td><td style="border: 1px solid white;">0.9056603773584906</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>cutoff</b></td><td style="border: 1px solid white;">0.33</td><td style="border: 1px solid white;">0.33</td><td style="border: 1px solid white;">0.33</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[11]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[3]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Multiclass Classification: Using automatic cutoffs</span>
<span class="n">model</span><span class="o">.</span><span class="n">classification_report</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Iris-setosa</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Iris-versicolor</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Iris-virginica</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>auc</b></td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">0.9958999999999999</td><td style="border: 1px solid white;">0.9959000000000002</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>prc_auc</b></td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">0.9922341855927179</td><td style="border: 1px solid white;">0.9920201374357384</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>accuracy</b></td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">0.96</td><td style="border: 1px solid white;">0.9666666666666667</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>log_loss</b></td><td style="border: 1px solid white;">0.00904411315578545</td><td style="border: 1px solid white;">0.0374738599818869</td><td style="border: 1px solid white;">0.0352009404192172</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>precision</b></td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">0.98</td><td style="border: 1px solid white;">1.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>recall</b></td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">0.9074074074074074</td><td style="border: 1px solid white;">0.9090909090909091</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>f1_score</b></td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">0.9467154769168191</td><td style="border: 1px solid white;">0.9523809523809523</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>mcc</b></td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">0.9133462590326239</td><td style="border: 1px solid white;">0.929320377284585</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>informedness</b></td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">0.8969907407407409</td><td style="border: 1px solid white;">0.9090909090909092</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>markedness</b></td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">0.9299999999999999</td><td style="border: 1px solid white;">0.95</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>csi</b></td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">0.8909090909090909</td><td style="border: 1px solid white;">0.9090909090909091</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>cutoff</b></td><td style="border: 1px solid white;">0.9</td><td style="border: 1px solid white;">0.407</td><td style="border: 1px solid white;">0.172</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[3]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[4]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Multiclass Classification: Customized Cutoffs</span>
<span class="n">model</span><span class="o">.</span><span class="n">classification_report</span><span class="p">(</span><span class="n">cutoff</span> <span class="o">=</span> <span class="p">[</span><span class="mf">0.8</span><span class="p">,</span> <span class="mf">0.4</span><span class="p">,</span> <span class="mf">0.2</span><span class="p">])</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Iris-setosa</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Iris-versicolor</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Iris-virginica</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>auc</b></td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">0.9958999999999999</td><td style="border: 1px solid white;">0.9959000000000002</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>prc_auc</b></td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">0.9922341855927179</td><td style="border: 1px solid white;">0.9920201374357384</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>accuracy</b></td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">0.96</td><td style="border: 1px solid white;">0.96</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>log_loss</b></td><td style="border: 1px solid white;">0.00904411315578545</td><td style="border: 1px solid white;">0.0374738599818869</td><td style="border: 1px solid white;">0.0352009404192172</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>precision</b></td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">0.98</td><td style="border: 1px solid white;">0.98</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>recall</b></td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">0.9074074074074074</td><td style="border: 1px solid white;">0.9074074074074074</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>f1_score</b></td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">0.9467154769168191</td><td style="border: 1px solid white;">0.9467154769168191</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>mcc</b></td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">0.9133462590326239</td><td style="border: 1px solid white;">0.9133462590326239</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>informedness</b></td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">0.8969907407407409</td><td style="border: 1px solid white;">0.8969907407407409</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>markedness</b></td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">0.9299999999999999</td><td style="border: 1px solid white;">0.9299999999999999</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>csi</b></td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">0.8909090909090909</td><td style="border: 1px solid white;">0.8909090909090909</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>cutoff</b></td><td style="border: 1px solid white;">0.8</td><td style="border: 1px solid white;">0.4</td><td style="border: 1px solid white;">0.2</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[4]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[5]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Multiclass Classification: Choosing the categories</span>
<span class="n">model</span><span class="o">.</span><span class="n">classification_report</span><span class="p">(</span><span class="n">labels</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;Iris-versicolor&quot;</span><span class="p">,</span> <span class="s2">&quot;Iris-virginica&quot;</span><span class="p">])</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Iris-versicolor</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Iris-virginica</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>auc</b></td><td style="border: 1px solid white;">0.9958999999999999</td><td style="border: 1px solid white;">0.9959000000000002</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>prc_auc</b></td><td style="border: 1px solid white;">0.9922341855927179</td><td style="border: 1px solid white;">0.9920201374357384</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>accuracy</b></td><td style="border: 1px solid white;">0.96</td><td style="border: 1px solid white;">0.9666666666666667</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>log_loss</b></td><td style="border: 1px solid white;">0.0374738599818869</td><td style="border: 1px solid white;">0.0352009404192172</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>precision</b></td><td style="border: 1px solid white;">0.98</td><td style="border: 1px solid white;">1.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>recall</b></td><td style="border: 1px solid white;">0.9074074074074074</td><td style="border: 1px solid white;">0.9090909090909091</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>f1_score</b></td><td style="border: 1px solid white;">0.9467154769168191</td><td style="border: 1px solid white;">0.9523809523809523</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>mcc</b></td><td style="border: 1px solid white;">0.9133462590326239</td><td style="border: 1px solid white;">0.929320377284585</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>informedness</b></td><td style="border: 1px solid white;">0.8969907407407409</td><td style="border: 1px solid white;">0.9090909090909092</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>markedness</b></td><td style="border: 1px solid white;">0.9299999999999999</td><td style="border: 1px solid white;">0.95</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>csi</b></td><td style="border: 1px solid white;">0.8909090909090909</td><td style="border: 1px solid white;">0.9090909090909091</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>cutoff</b></td><td style="border: 1px solid white;">0.407</td><td style="border: 1px solid white;">0.172</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[5]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[9]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Binary Classification</span>
<span class="n">model</span> <span class="o">=</span> <span class="n">RandomForestClassifier</span><span class="p">(</span><span class="n">name</span> <span class="o">=</span> <span class="s2">&quot;public.RF_titanic&quot;</span><span class="p">,</span>
                               <span class="n">n_estimators</span> <span class="o">=</span> <span class="mi">20</span><span class="p">,</span>
                               <span class="n">max_features</span> <span class="o">=</span> <span class="s2">&quot;auto&quot;</span><span class="p">,</span>
                               <span class="n">max_leaf_nodes</span> <span class="o">=</span> <span class="mi">32</span><span class="p">,</span> 
                               <span class="n">sample</span> <span class="o">=</span> <span class="mf">0.7</span><span class="p">,</span>
                               <span class="n">max_depth</span> <span class="o">=</span> <span class="mi">3</span><span class="p">,</span>
                               <span class="n">min_samples_leaf</span> <span class="o">=</span> <span class="mi">5</span><span class="p">,</span>
                               <span class="n">min_info_gain</span> <span class="o">=</span> <span class="mf">0.0</span><span class="p">,</span>
                               <span class="n">nbins</span> <span class="o">=</span> <span class="mi">32</span><span class="p">)</span>
<span class="n">model</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="s2">&quot;public.titanic&quot;</span><span class="p">,</span> <span class="p">[</span><span class="s2">&quot;age&quot;</span><span class="p">,</span> <span class="s2">&quot;fare&quot;</span><span class="p">,</span> <span class="s2">&quot;sex&quot;</span><span class="p">],</span> <span class="s2">&quot;survived&quot;</span><span class="p">)</span>
<span class="c1"># Binary Classification: the cutoff is the probability</span>
<span class="c1"># to accept the class 1</span>
<span class="n">model</span><span class="o">.</span><span class="n">classification_report</span><span class="p">(</span><span class="n">cutoff</span> <span class="o">=</span> <span class="mf">0.5</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>value</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>auc</b></td><td style="border: 1px solid white;">0.8334002663228425</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>prc_auc</b></td><td style="border: 1px solid white;">0.8077841814280875</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>accuracy</b></td><td style="border: 1px solid white;">0.7841365461847389</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>log_loss</b></td><td style="border: 1px solid white;">0.202111440275729</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>precision</b></td><td style="border: 1px solid white;">0.6828644501278772</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>recall</b></td><td style="border: 1px solid white;">0.7458100558659218</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>f1_score</b></td><td style="border: 1px solid white;">0.774572607363175</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>mcc</b></td><td style="border: 1px solid white;">0.5418686749886885</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>informedness</b></td><td style="border: 1px solid white;">0.551452689094762</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>markedness</b></td><td style="border: 1px solid white;">0.5324512269873813</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>csi</b></td><td style="border: 1px solid white;">0.553941908713693</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>cutoff</b></td><td style="border: 1px solid white;">0.5</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[9]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[10]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Binary Classification: automatic cutoff</span>
<span class="n">model</span><span class="o">.</span><span class="n">classification_report</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>value</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>auc</b></td><td style="border: 1px solid white;">0.8334002663228425</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>prc_auc</b></td><td style="border: 1px solid white;">0.8077841814280875</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>accuracy</b></td><td style="border: 1px solid white;">0.7911646586345381</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>log_loss</b></td><td style="border: 1px solid white;">0.202111440275729</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>precision</b></td><td style="border: 1px solid white;">0.7416879795396419</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>recall</b></td><td style="border: 1px solid white;">0.7304785894206549</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>f1_score</b></td><td style="border: 1px solid white;">0.777672475068387</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>mcc</b></td><td style="border: 1px solid white;">0.5633444041046916</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>informedness</b></td><td style="border: 1px solid white;">0.5618642321585514</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>markedness</b></td><td style="border: 1px solid white;">0.5648284754074107</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>csi</b></td><td style="border: 1px solid white;">0.5823293172690763</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>cutoff</b></td><td style="border: 1px solid white;">0.325</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[10]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;</pre>
</div>

</div>

</div>
</div>

</div>