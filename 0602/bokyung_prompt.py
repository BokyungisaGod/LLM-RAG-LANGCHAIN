# dustmq
prompt = """
system:
당신의 역할은 {question}의 질문을 {context}에서 찾아 답변해야 합니다.

context:  
''''''하이브와 민희진 어도어 대표 간의 갈등이 1개월 넘게 이어지는 가운데 뉴진스 팬 1만명이 민 대표의 해임을 반대하는 취지의 탄원서를 법원에 제출했다. 24일 가요계에 따르면 '버니즈'(뉴진스 팬덤) 1만명은 이날 오후 3시께 서울중앙지법 제50민사부에 탄원서를 냈다. 팬들은 탄원서에서 '민 대표가 위법한 행동을 했다는 것이 법적으로 최종 결론이 나기 전까지는 당사자 사이의 계약 내용은 존중돼야 하고, 그때까지 민 대표의 어도어 대표이사 지위가 유지되기를 희망한다는 것이 뉴진스 멤버들의 뜻임을 저희는 잘 알고 있다'며 '뉴진스를 지원하는 저희의 뜻 또한 마찬가지'라고 썼다. 전날 이 탄원서 서명이 시작된 이후 약 16시간 만에 팬들이 목표로 한 서명 참여자 1만명이 채워졌다. 하이브는 민 대표의 경영권 탈취 시도를 제기하며 대표이사 해임을 추진하고 있다. 이를 위한 어도어 임시주주총회는 오는 31일로 예정됐다. 민 대표는 이에 맞서 법원에 의결권 행사 금지 가처분 신청을 낸 상태다. 가처분 신청 결과는 다음 주중 임시주총 이전에 나올 전망이다.''''''

question: 탄원서에 뭐라고 적혀있대?
# """
# question에 본문과 관련없는 내용에 대해 질문하면 gpt가 제대로 된 답변을 하지 못함(역할을 잊어버림)