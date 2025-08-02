KeyError: This app has encountered an error. The original error message is redacted to prevent data leaks. Full error details have been recorded in the logs (if you're on Streamlit Cloud, click on 'Manage app' in the lower right of your app).
Traceback:
File "/mount/src/quickcast/app.py", line 71, in <module>
    forecast_df, kpi_df, error = run_all_models(sku_df, forecast_periods, freq)
                                 ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/mount/src/quickcast/utils/forecast_engine.py", line 24, in run_all_models
    df = sku_df.groupby("Period").agg({"Quantity": "sum"}).reset_index()
         ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.13/site-packages/pandas/core/groupby/generic.py", line 1432, in aggregate
    result = op.agg()
File "/home/adminuser/venv/lib/python3.13/site-packages/pandas/core/apply.py", line 190, in agg
    return self.agg_dict_like()
           ~~~~~~~~~~~~~~~~~~^^
File "/home/adminuser/venv/lib/python3.13/site-packages/pandas/core/apply.py", line 423, in agg_dict_like
    return self.agg_or_apply_dict_like(op_name="agg")
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.13/site-packages/pandas/core/apply.py", line 1603, in agg_or_apply_dict_like
    result_index, result_data = self.compute_dict_like(
                                ~~~~~~~~~~~~~~~~~~~~~~^
        op_name, selected_obj, selection, kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
File "/home/adminuser/venv/lib/python3.13/site-packages/pandas/core/apply.py", line 462, in compute_dict_like
    func = self.normalize_dictlike_arg(op_name, selected_obj, func)
File "/home/adminuser/venv/lib/python3.13/site-packages/pandas/core/apply.py", line 663, in normalize_dictlike_arg
    raise KeyError(f"Column(s) {list(cols)} do not exist")
