import pandas as pd
import pyodbc
import os
from sqlalchemy import create_engine, NVARCHAR


def merge_excel_to_sqlserver(folder_path, table_name):
    """
    合并Excel文件并导入SQL Server

    参数:
        folder_path (str): 包含Excel文件的文件夹路径
        server (str): SQL Server地址
        database (str): 数据库名称
        table_name (str): 目标表名
    """
    output_file = folder_path + "\\合并结果.xlsx"  # 输出文件名
    try:
        # 第一步：合并所有Excel文件
        print("开始合并Excel文件...")
        all_data = []

        # 遍历文件夹中的所有Excel文件
        for file in os.listdir(folder_path):
            if file.endswith(('.xlsx', '.xls')) and not file.endswith(output_file):
                file_path = os.path.join(folder_path, file)
                print(f"正在处理文件: {file}")

                # 读取Excel文件中的所有sheet
                excel_data = pd.read_excel(file_path, sheet_name=None)

                # 合并所有sheet到一个DataFrame
                for sheet_name, df in excel_data.items():
                    df['source_file'] = file  # 添加来源文件列
                    df['sheet_name'] = sheet_name  # 添加sheet名称列
                    all_data.append(df)

        if not all_data:
            print("未找到任何Excel文件！")
            return False

        # 合并所有DataFrame
        merged_df = pd.concat(all_data, ignore_index=True)
        print(f"合并完成，总记录数: {len(merged_df)}")
        merged_df.to_excel(output_file, index=False)
        print(f"成功合并{len(all_data)}个文件，已保存为: {output_file}")
        # 第二步：连接到SQL Server并导入数据
        print("开始导入SQL Server...")

        # 创建SQL Server连接字符串
        # 使用Windows身份验证
        connection_string = (
            "mssql+pyodbc://@localhost/HearingTestingData?"
            "driver=ODBC+Driver+17+for+SQL+Server&"
            "trusted_connection=yes"
            "charset=utf8"
        )

        # 获取包含中文的列名
        chinese_columns = [col for col in df.columns if df[col].astype(str).str.contains(r'[\u4e00-\u9fff]').any()]

        # 为中文列创建类型映射
        dtype = {col: NVARCHAR(255) for col in chinese_columns}
        # 创建SQLAlchemy引擎
        engine = create_engine(connection_string)

        # 导入数据到SQL Server
        merged_df.to_sql(
            name=table_name,
            con=engine,
            if_exists='replace',  # 如果表存在则替换，可选'append'追加
            index=False,
            chunksize=1000,  # 分批提交，提高大文件导入性能
            dtype=dtype
        )

        print(f"数据成功导入到SQL Server表 {table_name}")
        return True

    except Exception as e:
        print(f"发生错误: {str(e)}")
        return False


# 使用示例
if __name__ == "__main__":
    # 配置参数
    excel_folder = r"D:\Downloads\TestData"  # Excel文件所在文件夹

    target_table = "hearingtestdata"  # 目标表名

    # 执行合并和导入
    success = merge_excel_to_sqlserver(
        excel_folder,
        target_table
    )

    if success:
        print("操作成功完成！")
    else:
        print("操作失败，请检查错误信息。")