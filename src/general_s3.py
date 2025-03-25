import os
import yaml
import textwrap
import json
import re
import itertools
from sqlalchemy.sql import text
from sqlalchemy import exc # exceptions
from sqlalchemy import create_engine as _create_engine
import boto3
from botocore.exceptions import ClientError
from botocore.config import Config
from boto3.s3.transfer import S3Transfer
import urllib.parse
from io import BytesIO, StringIO
import pickle


"""
This code uses a json file to store database credentials for simplicity.
AWS credentials are similarly stored in default .aws location.
"""
home = os.path.expanduser("~")
DB_CREDENTIAL_FILE = os.path.join(home, '.db_utils', 'db_creds.json')
DEFAULT_REDSHIFT_CONNECTION_NAME = 'default_redshift'


def _create_db_engine(db_type, host, port, database, user, password, connect_args=[]):
    # special characters such as those that may be used in the password need to be URL encoded to be parsed correctly
    return _create_engine(db_type + '://' + user + ':' + urllib.parse.quote_plus(password) +
                          '@' + host + ':' + str(port) + '/' + database,
                          connect_args=connect_args)


def _create_db_engine_with_credentials(db_cred_file_name, connection_name, connect_args=[]):
    with open(db_cred_file_name) as json_file:
        db_credentials = json.load(json_file)
    conn = db_credentials[connection_name]
    return _create_db_engine(conn['db_type'], conn['host'], conn['port'], conn['database'],
                             conn['user'], conn['password'], connect_args=connect_args)


def create_redshift_engine(db_cred_file=DB_CREDENTIAL_FILE, connection_name=DEFAULT_REDSHIFT_CONNECTION_NAME, connect_args={}):
    """
    Create a connection engine for the Redshift DB using the specs from
    the specified db_cred_file.

    Keyword arguments:
    db_cred_file -- defaults to creds stored in local directory loaded on package
                    import, but can be overridden. Credential file must match format
                    and naming conventions of db_creds.json.example
    connect_args -- allows for extra parameters when establishing the DB connection.
                    This used to include {'keepalives': '1', 'keepalives_idle': '60'}
                    by default.
    """
    default_connect_args = {'sslmode': 'prefer'} # needed because of issues with SSL
    connect_args.update(default_connect_args)

    return _create_db_engine_with_credentials(db_cred_file, connection_name, connect_args=connect_args)


def _redshift_to_df(_query):
    redshift_engine = create_redshift_engine(connection_name='default_redshift')
    try:
        result = pd.read_sql_query(_query, redshift_engine)
    except TypeError as e:
        _msg = "{} - Check query for errors - '%' character is not valid in current query format".format(e)
        logger.error(_msg)
        return
    except (exc.DBAPIError, Exception) as e:
        exc_buffer = StringIO()
        traceback.print_exc(file=exc_buffer)
        # print/send exception traceback log, etc.
        print(exc_buffer.getvalue())
        if e.connection_invalidated:
            redshift_engine.dispose()
            redshift_engine = create_redshift_engine(connection_name='default_redshift')
            result = pd.read_sql_query(_query, redshift_engine)
        else:
            raise e
    redshift_engine.dispose()
    return result


def _datalake_to_df(_query):
    redshift_engine = create_redshift_engine(connection_name='default_redshift')
    try:
        result = pd.io.sql.read_sql(_query, redshift_engine)
    except exc.DBAPIError as e:
        exc_buffer = StringIO()
        traceback.print_exc(file=exc_buffer)
        # print/send exception traceback log, etc.
        print(exc_buffer.getvalue())
        if e.connection_invalidated:
            redshift_engine.dispose()
            redshift_engine = create_redshift_engine(connection_name='default_redshift')
            result = pd.io.sql.read_sql(_query, redshift_engine)
    redshift_engine.dispose()
    return result


def _s3_client(profile='default'):
    session = boto3.Session(profile_name=profile)
    return session.client('s3')


def _to_s3(data, key, client, bucket):
    # assumes that ACL is enabled
    acl = "bucket-owner-full-control"
    to_write = data.to_csv(None,index=False).encode()
    response = client.put_object(Body=to_write,Bucket=bucket,Key=key,ACL=acl)
    return response


def _png_to_s3(img, key, client, bucket):
    # assumes that ACL is enabled
    acl = "bucket-owner-full-control"
    response = client.put_object(Body=img.getvalue(),Bucket=bucket,Key=key,ContentType='image/png',ACL=acl)
    return response


def _df_parquet_to_s3(df,bucket,client,key,prefix):
    # assumes that ACL is enabled
    acl = "bucket-owner-full-control"
    buffer = BytesIO()
    df.to_parquet(buffer)
    response = client.put_object(Body=buffer.getvalue(),Bucket=bucket,Key='/'.join([prefix,key]),ACL=acl)
    if response['ResponseMetadata']['HTTPStatusCode'] != 200:
        print(response['ResponseMetadata'])
    return response['ResponseMetadata']


def _df_parquet_from_s3(bucket,client,key,prefix):
    buffer = BytesIO()
    client.download_fileobj(bucket,'/'.join([prefix,key]),buffer)
    return pd.read_parquet(buffer)


def _pickle_dump_to_s3(pkl,bucket,client,key,prefix):
    # assumes that ACL is enabled
    acl = "bucket-owner-full-control"
    buffer = BytesIO()
    pickle.dump(pkl,buffer)
    response = client.put_object(Body=buffer.getvalue(),Bucket=bucket,Key='/'.join([prefix,key]),ACL=acl)
    return response


def _pickle_load_from_s3(bucket,client,key,prefix):
    buffer = BytesIO()
    try:
        client.download_fileobj(bucket,'/'.join([prefix,key]),buffer)
    except client.exceptions.ClientError as e:
        _msg = f'No such model exists - {bucket}/{prefix}/{key}'
        print(_msg)
        raise
    with open('/tmp/temp.pkl','wb') as f:
        f.write(buffer.getvalue())
    with open('/tmp/temp.pkl','rb') as f:
        pkl = pickle.load(f)
    return pkl


def _s3_key_list(s3_bucket, prefix, pattern, profile='default'):
    # return list of keys with regex filter
    client = _s3_client(profile)
    _pattern = re.compile(pattern)
    contents = []
    kwargs = {'Bucket':s3_bucket,'Prefix':prefix}
    while True:
        resp = client.list_objects_v2(**kwargs)
        try:
            contents.extend(resp['Contents'])
            kwargs['ContinuationToken'] = resp['NextContinuationToken']
        except KeyError:
            break
    keylist = [f['Key'] for f in contents if _pattern.search(f['Key']) is not None]
    return keylist


def _csv_to_redshift(data,client,bucket):
    # save data to s3 as .csv then copy to redshift
    file = 'temp.csv'
    r = _to_s3(data, file, client, bucket)
    sess = boto3._get_default_session()
    key, secret = sess.get_credentials().access_key, sess.get_credentials().secret_key
    redshift_engine = create_redshift_engine(connection_name='default_redshift')
    # create temp table that matches destination table schema
    create = """
             CREATE TABLE tmp_table
             (
                distkey_data type distkey,
                primary_key type,
                sortkey_1 type,
                sortkey_2 type,
                data_1 type,
                data_2 type,
                data_3 type,
                data_4 type
             )
             SORTKEY (sortkey_1, sortkey_2)
             """
    # populate temp table with data from .csv file
    stmt = """
           COPY tmp_table
               FROM 's3://{bucket}/{file}'
               CREDENTIALS 'aws_access_key_id={key};aws_secret_access_key={secret}'
               DELIMITER AS ','
               CSV IGNOREHEADER 1;
           """
    # remove existing data with same primary key to avoid duplication
    stmt2 = """
            DELETE FROM dest_schema.dest_table
            WHERE primary_key IN
                (
                SELECT primary_key
                FROM tmp_table
                )
            """
    with redshift_engine.begin() as conn:
        conn.execute("DROP TABLE IF EXISTS tmp_table;")
        conn.execute(create)
        conn.execute("ANALYZE tmp_table;")
        conn.execute(stmt.format(bucket=bucket,file=file,key=key,secret=secret))
        conn.execute(stmt2)
        conn.execute("INSERT INTO dest_schema.dest_table SELECT * FROM tmp_table;")
        conn.execute("DROP TABLE IF EXISTS tmp_table;")
        conn.execute("commit;")
    redshift_engine.dispose()


def _regex_filter(_string,_list):
    """
    filter a list of strings to only those which match a regex pattern
    INPUTS
        _string - str, regex pattern to match
        _list - list-like, list of string values to filter
    OUTPUT
        filtered list
    """
    return list(filter(re.compile(_string).search,_list))


def _quarter_start_end(data_date):
    # extract year and month from data_date
    if type(data_date) is str:
        _t0 = dt.strptime(data_date,'%Y-%m-%d')
        _y, _m = _t0.year, _t0.month
    else:
        _y, _m = data_date.year, data_date.month
    # convert month to calendar quarter
    _q = (_m - 1) // 3 + 1
    # get first day of calendar quarter
    current_quarter = f'{_y}-{3 * (_q - 1) + 1}-1'
    t0 = dt.strptime(current_quarter,'%Y-%m-%d').date()
    # generate a date in next quarter and extract year/month
    _t1 = t0 + td(days=100)
    n_y, n_m = _t1.year, _t1.month
    # convert to next quarter
    n_q = (n_m - 1) // 3 + 1
    next_quarter = f'{n_y}-{3 * (n_q - 1) + 1}-1'
    # subtract one day to get last day of current quarter
    _next_quarter = dt.strptime(next_quarter,'%Y-%m-%d')
    t1 = (_next_quarter - td(days=1)).date()
    # return first and last day of quarter as date
    return t0, t1
