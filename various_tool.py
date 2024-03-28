import json
from datetime import datetime

import requests
from langchain_core.tools import BaseTool


class OrderSearchTool(BaseTool):
    name = "订单查询"
    description = "查询订单信息，入参是订单编号（格式是DD加数字），多个订单号请用|分割，未提供订单号时请用「未提供」代替;"
    user_id: int
    session: str

    def _run(
            self,
            order_code: str,
    ) -> str:
        print(f'[OrderSearch][{datetime.now()}]用户订单查询，user_id={self.user_id},order_code={order_code}')

        try:
            if '未提供' in order_code:
                return '请以客服的角色和客户沟通他要咨询哪个订单，提供下订单号'
            base_url = 'https://api.example.com/order-search'
            params = {
                "userId": self.user_id,
                "orderCodeList": order_code.split("|")
            }
            response = requests.get(base_url, params=params)
            return response.json()['data']
        except BaseException as e:
            print(f'订单查询失败：{e}')
            return '抱歉，系统开小差啦，请稍后重试或者联系人工客服'


class OrderChangeTool(BaseTool):
    name = "订单修改"
    description = "修改订单信息，入参是订单编号（格式是DD加数字），多个订单号请用|分割，未提供订单号时请用「未提供」代替;"
    user_id: int
    session: str

    def _run(
            self,
            order_code: str,
    ) -> str:
        print(f'[OrderChangeTool][{datetime.now()}]用户订单修改，user_id={self.user_id},order_code={order_code}')

        try:
            if '未提供' in order_code:
                return '请以客服的角色和客户沟通他要修改哪个订单，提供下订单号'
            base_url = 'https://api.example.com/order-change'
            data = {
                "userId": self.user_id,
                "orderCodeList": order_code.split("|")
            }
            response = requests.post(base_url, params=json.dumps(data))
            return response.json()['data']
        except BaseException as e:
            print(f'订单修改失败：{e}')
            return '抱歉，系统开小差啦，请稍后重试或者联系人工客服'
