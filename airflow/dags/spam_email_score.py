from airflow import DAG
from airflow.operators.bash import BashOperator
import pendulum
import datetime as dt
 
args = {
    "owner": "admin",
    "start_date": dt.datetime(2024, 2, 5),
    "retries": 1,
    "retry_delays": dt.timedelta(minutes=1),
    "depends_on_past": False
}
 
with DAG(
    dag_id='spam_email_score',
    default_args=args,
    schedule_interval=None,
    tags=['spam', 'score'],
) as dag:
    get_data = BashOperator(task_id='get_data',
                            bash_command="python3 /home/lnc/project/scripts/get_data.py",
                            dag=dag)
    process_data = BashOperator(task_id='process_data',
                            bash_command="python3 /home/lnc/project/scripts/process_data.py",
                            dag=dag)
    train_test_split_data = BashOperator(task_id='train_test_split_data',
                            bash_command="python3 /home/lnc/project/scripts/train_test_split.py",
                            dag=dag)  
    train_model = BashOperator(task_id='train_model',
                            bash_command="python3 /home/lnc/project/scripts/train_model.py",
                            dag=dag)
    test_model = BashOperator(task_id='test_model',
                            bash_command="python3 /home/lnc/project/scripts/test_model.py",
                            dag=dag)
    get_data >> process_data >> train_test_split_data >> train_model >> test_model
