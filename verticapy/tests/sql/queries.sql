/* SQL Magic Test */
/* Drop a Model */
DROP MODEL IF EXISTS model_test;
/* Create a Model */
SELECT LINEAR_REG('model_test', 'public.titanic', 'survived', 'age, fare'); 
/* Compute a prediction */
SELECT PREDICT_LINEAR_REG(3.0, 4.0 
      USING PARAMETERS model_name='model_test', 
                       match_by_pos=True) AS predict;